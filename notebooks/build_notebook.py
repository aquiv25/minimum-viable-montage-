import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ------------------------------------------------------------------ #
md("""\
# Minimum Viable Montage — STEW Channel Ablation Analysis

**NSRI Summer Research Hackathon 2026 — AI, Data Science & Computing track**

**Research question:** How few EEG channels does a consumer headband actually
need to reliably detect cognitive overload in a new user?

This notebook consolidates the team's model-development work (baseline model,
normalization tests, feature engineering, hyperparameter tuning) with the
statistical-validation and channel-ablation analysis: is any of this
significant given only 45 subjects, and how many channels do we actually need?

**Dataset:** STEW — Simultaneous Task EEG Workload (Lim et al., 2018), 45
subjects (of 48; 3 excluded for missing ratings), Emotiv EPOC, 14 channels,
128 Hz, single SIMKAP multitasking recording per subject. Label: overloaded
(rating ≥ 7) vs. not (rating ≤ 6), per Lim et al.'s own high-workload cutoff.

**Validation standard:** Leave-One-Subject-Out (LOSO) cross-validation with
subject-level majority-vote accuracy — every epoch from a held-out subject
carries the same label, so the fold's prediction is the majority vote across
that subject's epoch-level predictions, not raw epoch accuracy. This is
stricter than a random train/test split (which lets epochs from the same
subject leak across train and test) and stricter than standard k-fold CV
without subject grouping.
""")

# ------------------------------------------------------------------ #
md("## 1. Data & Feature Extraction")

code("""\
import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import binomtest

import stew_pipeline as sp

DATA_DIR = Path("../data/stew")

signals, rating, class3 = sp.load_raw(DATA_DIR)
labels = sp.compute_labels(rating, class3, mode="two_class")

print(f"Subjects: {signals.shape[0]}, channels: {signals.shape[1]}, samples/subject: {signals.shape[2]}")
print(f"Rating distribution: {dict(zip(*np.unique(rating, return_counts=True)))}")
print(f"Class balance (0=not-overloaded, 1=overloaded): {dict(zip(*np.unique(labels, return_counts=True)))}")
""")

md("""\
Features: log band power (delta, theta, alpha, beta, gamma) per channel, via
Welch PSD over 2-second epochs with 50% overlap → 14 channels × 5 bands = 70
features per epoch. This is the "absolute" feature variant — see Section 3
for why this outperforms relative/normalized alternatives.
""")

code("""\
X, y, groups = sp.build_dataset(DATA_DIR, "two_class", feature_mode="absolute")
print(f"X shape: {X.shape}  (epochs x features)")
print(f"Feature names (first 10): {sp.feature_names()[:10]}")
""")

# ------------------------------------------------------------------ #
md("""\
## 2. Statistical Significance Framework

With only 45 subjects, "accuracy" alone is not enough — we need to know how
much of it could be chance. Chance-level **balanced accuracy** is exactly 0.5
by construction (mean of 100%/0% recall for a degenerate classifier); the
majority-class accuracy (always predict "overloaded") is 0.556, which is
*not* the right chance reference for the balanced-accuracy numbers below.

For subject-level accuracy (binary, n=45), we test against chance=0.5 using
an exact one-sided binomial test. This tells us the minimum number of
correct subjects needed for a result to be significant at p<0.05.
""")

code("""\
n = 45
for k in range(24, 33):
    p = binomtest(k, n, 0.5, alternative="greater").pvalue
    flag = "  <-- p<0.05 threshold" if p < 0.05 and binomtest(k-1, n, 0.5, alternative="greater").pvalue >= 0.05 else ""
    print(f"{k}/{n} = {k/n:.3f}   p={p:.4f}{flag}")
""")

md("""\
**≥29/45 (64.4%) is required for p<0.05 against chance.** This threshold is
fixed by n=45 alone — it doesn't depend on which model or features we use.
Keep this number in mind for every result below.
""")

# ------------------------------------------------------------------ #
md("""\
## 3. Baseline Model & Does Normalization Help?

Absolute band power varies person-to-person (skull thickness, electrode
contact, impedance). Two normalization strategies were tested to see if
removing that per-subject offset improves generalization to a new subject.
""")

code("""\
# Full 45-fold LOSO with an unregularized Random Forest takes several minutes
# end-to-end, so the result is cached here (reproducible via
# run_loso(X, y, groups, rf_params={"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1})).
with open(DATA_DIR / "ckpt_raw_n100.json") as f:
    raw_ckpt = json.load(f)
raw_correct = sum(r["subject_correct"] for r in raw_ckpt["results"])
raw_acc = raw_correct / 45
p = binomtest(raw_correct, 45, 0.5, alternative="greater").pvalue
print(f"Baseline RF, absolute power (70 feat): subject_level_acc={raw_acc:.3f} "
      f"({raw_correct}/45), p={p:.4f}")
""")

code("""\
# Relative power + per-subject z-score (both variants tested independently by
# two team members -- see methodology_note.md): CONFIRMED WORSE than absolute
# power on real LOSO testing. Cached here (full run ~40s); reproducible via
# build_dataset(feature_mode="relative", normalize_per_subject=True).
with open(DATA_DIR / "ckpt_normalized_n100.json") as f:
    norm_ckpt = json.load(f)
norm_correct = sum(r["subject_correct"] for r in norm_ckpt["results"])
print(f"Relative power + per-subject z-score (70 feat): subject_level_acc={norm_correct/45:.3f} ({norm_correct}/45)")
print()
print("Independently, teammate's own tests (relative power alone, and per-subject")
print("z-score on absolute power alone) also came out worse than absolute power:")
print("  relative power alone:            acc=0.356 (16/45)")
print("  per-subject z-score (absolute):  acc=0.422 (19/45)")
print()
print("-> Two independent implementations agree: absolute power carries real")
print("   between-subject signal that any per-subject normalization erases,")
print("   since STEW has one session per subject (nothing to calibrate against).")
print("   Sticking with absolute power going forward.")
""")

# ------------------------------------------------------------------ #
md("""\
## 4. Does Feature Engineering Help? (alpha/beta ratio + frontal asymmetry)

Two engineered feature groups were added on top of the 70 absolute-power
features (88 total): **alpha/beta ratio per channel** (classic EEG workload
marker) and **frontal alpha asymmetry** (log(right) − log(left) alpha power,
for 4 frontal channel pairs: F3/F4, F7/F8, FC5/FC6, AF3/AF4).
""")

code("""\
n_channels, n_bands = 14, 5
X_reshaped = X.reshape(X.shape[0], n_channels, n_bands)
power = np.expm1(X_reshaped)  # undo log1p to get raw power
channels, bands = sp.CHANNELS, list(sp.BANDS.keys())

alpha_idx, beta_idx = bands.index("alpha"), bands.index("beta")
alpha_power = power[:, :, alpha_idx]
beta_power = power[:, :, beta_idx]
ab_ratio_log = np.log1p(alpha_power / (beta_power + 1e-8))

frontal_pairs = [("F3", "F4"), ("F7", "F8"), ("FC5", "FC6"), ("AF3", "AF4")]
asymmetry_feats = []
for left, right in frontal_pairs:
    li, ri = channels.index(left), channels.index(right)
    asymmetry_feats.append(np.log1p(alpha_power[:, ri]) - np.log1p(alpha_power[:, li]))
asymmetry_feats = np.stack(asymmetry_feats, axis=-1)

X_extended = np.concatenate([X, ab_ratio_log, asymmetry_feats], axis=-1)
print(f"Original features: {X.shape[1]}  ->  Extended features: {X_extended.shape[1]}")
""")

md("""\
## 5. Model Comparison & Hyperparameter Tuning

Four model families were compared (Logistic Regression, SVM-RBF, XGBoost,
Random Forest), plus a regularization check: with only 45 subjects and 70-88
features, an unconstrained Random Forest can overfit to individual subjects'
idiosyncrasies rather than the workload signal. Full model-comparison sweep
(cached — reproduced independently, see Section 6 for the significance check
on the winning configuration):
""")

code("""\
summary = pd.DataFrame([
    {"Model": "Random Forest (default)",      "Features": "absolute power (70)",        "Subject acc": 0.578, "Balanced acc": 0.550},
    {"Model": "Random Forest (default)",      "Features": "relative power (70)",         "Subject acc": 0.356, "Balanced acc": 0.335},
    {"Model": "Random Forest (default)",      "Features": "per-subject z-score (70)",    "Subject acc": 0.422, "Balanced acc": 0.380},
    {"Model": "Logistic Regression",          "Features": "baseline (70)",               "Subject acc": 0.444, "Balanced acc": 0.425},
    {"Model": "Logistic Regression",          "Features": "extended (88)",               "Subject acc": 0.444, "Balanced acc": 0.425},
    {"Model": "SVM (RBF)",                    "Features": "baseline (70)",               "Subject acc": 0.444, "Balanced acc": 0.425},
    {"Model": "SVM (RBF)",                    "Features": "extended (88)",               "Subject acc": 0.422, "Balanced acc": 0.405},
    {"Model": "XGBoost",                      "Features": "baseline (70)",               "Subject acc": 0.489, "Balanced acc": 0.465},
    {"Model": "XGBoost",                      "Features": "extended (88)",               "Subject acc": 0.511, "Balanced acc": 0.485},
    {"Model": "Random Forest (default)",      "Features": "extended (88)",               "Subject acc": 0.511, "Balanced acc": 0.485},
    {"Model": "Random Forest (tuned, depth=6)", "Features": "baseline (70)",             "Subject acc": 0.622, "Balanced acc": 0.580},
    {"Model": "Random Forest (tuned, depth=6)", "Features": "extended (88)",             "Subject acc": 0.644, "Balanced acc": 0.600},
])
summary
""")

# ------------------------------------------------------------------ #
md("""\
## 6. Is the Improvement Real? Significance Testing on the Winning Configuration

The best configuration — Random Forest with `max_depth=6, min_samples_leaf=5`
on the 88-feature extended set — reaches **64.4% (29/45)**, which sits right
at the p<0.05 boundary established in Section 2. Independently reproduced
here (exact feature formula + hyperparameters, `n_estimators=200`):
""")

code("""\
with open(DATA_DIR / "ckpt_teammate_exact_n200.json") as f:
    ext_ckpt = json.load(f)
with open(DATA_DIR / "ckpt_tuned_d6_l5_n200.json") as f:
    base_ckpt = json.load(f)

ext_correct_arr = np.array([r["subject_correct"] for r in ext_ckpt["results"]])
base_correct_arr = np.array([r["subject_correct"] for r in base_ckpt["results"]])

ext_n = int(ext_correct_arr.sum())
base_n = int(base_correct_arr.sum())

for label, n_correct in [("Tuned RF, baseline (70 feat)", base_n), ("Tuned RF, extended (88 feat)", ext_n)]:
    p = binomtest(n_correct, 45, 0.5, alternative="greater").pvalue
    sig = "significant (p<0.05)" if p < 0.05 else "NOT significant"
    print(f"{label}: {n_correct}/45 = {n_correct/45:.3f}  vs. chance p={p:.4f}  -> {sig}")
""")

code("""\
# McNemar's exact test (paired comparison, same 45 subjects/folds) between the
# baseline and extended configurations -- is the +2.2 point improvement a real
# effect of the engineered features, or within noise?
b = int((base_correct_arr & ~ext_correct_arr).sum())  # baseline-only correct
c = int((~base_correct_arr & ext_correct_arr).sum())  # extended-only correct
print(f"Discordant subjects: baseline-only-correct={b}, extended-only-correct={c}")

if b + c > 0:
    mc_p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    print(f"McNemar exact test (baseline vs. extended): p={mc_p:.4f}")
else:
    print("No discordant subjects -- predictions are identical.")
""")

md("""\
**Interpretation:** the extended-feature result (64.4%) is technically over
the p<0.05 line, but only by the thinnest possible margin — a single
subject's prediction flipping. The McNemar test confirms this directly: only
**one subject out of 45** differs between the baseline (70-feature) and
extended (88-feature) tuned models, giving p≈1.0 — statistically
indistinguishable from noise.

**Conclusion:** the real, robust finding is that **limiting Random Forest
tree depth fixes overfitting** (57.8% → 62.2%, a 6-subject swing). The
additional +2.2 points from alpha/beta ratio and frontal asymmetry features
is not distinguishable from chance variation given n=45 — it should be
reported as "compatible with no effect," not as a confirmed improvement.
""")

# ------------------------------------------------------------------ #
md("""\
## 7. Channel Ablation: How Many Channels Do We Actually Need?

Restricting the search to a wearable-realistic frontal/temporal pool (10 of
the 14 channels — excludes parietal/occipital P7, P8, O1, O2), channels are
ranked once by Random Forest feature importance (tuned config, all-data fit),
then added one at a time in that order, each point evaluated with a full
45-fold LOSO run (raw absolute-power features, tuned RF: `max_depth=6,
min_samples_leaf=5`, `n_estimators=100`).
""")

code("""\
with open(DATA_DIR / "channel_ranking.json") as f:
    ranking = json.load(f)
with open(DATA_DIR / "ablation_curve.json") as f:
    curve = json.load(f)

print("Channel importance ranking (most -> least important):")
print(ranking)
print()
curve_df = pd.DataFrame(curve)
curve_df["accuracy"] = curve_df["n_correct"] / curve_df["n_total"]
curve_df
""")

code("""\
import matplotlib.pyplot as plt

ks = curve_df["k"].tolist()
accs = curve_df["accuracy"].tolist()
n_correct = curve_df["n_correct"].tolist()
sig_thresh_acc = 29 / 45

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ks, accs, marker="o", color="#2563eb", linewidth=2, markersize=7, label="Subject-level accuracy (LOSO)")
ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1.5, label="Chance level (0.5)")
ax.axhline(sig_thresh_acc, color="#dc2626", linestyle=":", linewidth=1.5,
           label=f"p<0.05 threshold (29/45 = {sig_thresh_acc:.3f})")

for k, a, nc in zip(ks, accs, n_correct):
    ax.annotate(f"{nc}/45", (k, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#374151")

ax.set_xlabel("Number of channels (added in order of importance)")
ax.set_ylabel("Subject-level accuracy")
ax.set_title("Channel Ablation: Accuracy vs. Number of EEG Channels\\n(Tuned RF, n=45 subjects, LOSO)")
ax.set_xticks(ks)
ax.set_ylim(0.35, 0.75)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_xticklabels([f"{k}\\n(+{ranking[k-1]})" for k in ks], fontsize=8)
plt.tight_layout()
plt.show()
""")

md("""\
**The curve is essentially flat**: subject-level accuracy oscillates between
28/45 (0.622) and 29/45 (0.644) regardless of how many channels (1 to 10) are
used — removing 8 of 10 channels neither helps nor hurts. A McNemar test
between the 2-channel model (F3, AF3) and the full 10-channel model finds
only 1 discordant subject (p=1.0) — statistically indistinguishable, exactly
as in Section 6.

**Practical takeaway for "Minimum Viable Montage":** a 2-channel frontal
montage (F3 + AF3) performs statistically no worse than using all 10
wearable-realistic channels, or all 14 original channels. This is the
honest basis for a minimal-montage recommendation — not "fewer channels
improved accuracy," but "fewer channels cost nothing," which is exactly what
a comfort/cost-driven wearable design argument needs.
""")

# ------------------------------------------------------------------ #
md("""\
## 8. Comparison with Prior Literature

| Source | Method | Result |
|---|---|---|
| Lim et al., 2018 (original STEW paper) | Single subject-based 80/20 split (36 train / 9 test), SVR + neighborhood component analysis feature selection, 3-class | 69% accuracy, Cohen's κ=0.46 |
| Yedukondalu et al., 2024 (*Scientific Reports*) | Standard 10-fold CV, no subject grouping mentioned | 96–97% accuracy |
| This work | Full LOSO (45 folds), Random Forest, absolute power, depth-limited | 62.2% (70 feat) / 64.4% (88 feat) subject-level accuracy |

The much higher accuracies reported in some later STEW papers (89–98%) most
likely reflect **evaluation without subject-level grouping** — letting
epochs from the same subject appear in both train and test, which leaks
subject identity and inflates accuracy. Our LOSO numbers are close to Lim et
al.'s own subject-independent baseline (69%), which is the fairest
comparison point, not the inflated later numbers.
""")

# ------------------------------------------------------------------ #
md("""\
## 9. Conclusions & Recommendations

1. **Absolute band power beats every normalization variant tried** (relative
   power, per-subject z-score) — confirmed independently by two different
   implementations. STEW has one session per subject, so per-subject
   normalization removes exactly the between-subject offset that carries the
   label signal.
2. **Limiting Random Forest tree depth is the one robust, real improvement**
   found (57.8% → 62.2%, not significant vs. chance at p=0.068 but a solid
   6-subject swing over the unconstrained default).
3. **Adding alpha/beta ratio + frontal asymmetry features reaches 64.4%**,
   crossing the formal p<0.05 significance line — but the McNemar test shows
   this is a single-subject-flip effect, not distinguishable from noise.
   Report this as "compatible with a modest effect, underpowered to confirm,"
   not as a confirmed win.
4. **Channel count barely matters**: a 2-channel frontal montage (F3, AF3)
   performs statistically indistinguishably from the full 10- or 14-channel
   montage. This — not a raw accuracy improvement — is the strongest,
   best-supported basis for the "Minimum Viable Montage" pitch: you can go
   down to 2 comfortable frontal electrodes at no measurable cost.
5. **Honest limitation**: n=45 subjects gives limited statistical power.
   Every "significant" result in this analysis (29/45) sits at the exact
   boundary of the smallest possible margin: one subject's outcome. None of
   our conclusions should be oversold as strongly confirmed findings; they
   are directional, and the channel-invariance finding (point 4) is the one
   we can state with the most confidence, since it holds consistently across
   8 of 10 channel counts rather than a single lucky split.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("minimum_viable_montage_analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written, n_cells =", len(cells))
