"""
Honest evaluation: Leave-One-Subject-Out (LOSO) cross-validation, plus the channel
ablation search (accuracy vs. number of channels, restricted to a wearable-realistic
frontal/temporal montage).

This is the result the project's headline claim should be based on -- not the
random-split baseline in stew_baseline.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, balanced_accuracy_score, f1_score,
    roc_auc_score,
)
from itertools import combinations

from stew_pipeline import DATA_DIR, feature_names, CHANNELS, FRONTAL_TEMPORAL, BANDS

# Balanced accuracy of ANY degenerate/majority-only classifier is exactly 0.5
# by construction (mean of 100% recall on one class, 0% on the other). That's
# the real chance-level reference point -- NOT the raw accuracy of an
# always-predict-majority-class rule (which was 0.556 here and, being on the
# accuracy scale, isn't comparable to balanced_accuracy / AUC numbers below).
CHANCE_LEVEL_BALANCED_ACC = 0.5

# Quick, non-nested regularization check (NOT a grid search / hyperparameter
# tuning with nested CV -- too expensive for the timeline). Just: does capping
# tree depth or requiring bigger leaves help, given 45 subjects / 70 features
# is a small-n regime where an unconstrained RF (max_depth=None) can overfit
# per-subject idiosyncrasies rather than the workload signal.
REGULARIZED_RF_CONFIGS = {
    "default (unregularized)": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    "max_depth=5": {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 1},
    "max_depth=10": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 1},
    "min_samples_leaf=10": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10},
    "max_depth=10, min_samples_leaf=5": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
}


def channel_feature_mask(selected_channels, all_channels=CHANNELS, bands=BANDS):
    """Boolean mask into the flat feature vector for a subset of channels.

    Feature order from stew_pipeline.band_power_features is
    [ch0_band0, ch0_band1, ..., ch1_band0, ...], so each channel occupies
    len(bands) consecutive slots.
    """
    n_bands = len(bands)
    mask = np.zeros(len(all_channels) * n_bands, dtype=bool)
    for ch in selected_channels:
        idx = all_channels.index(ch)
        mask[idx * n_bands:(idx + 1) * n_bands] = True
    return mask


def run_loso(X, y, groups, channel_mask=None, rf_params=None):
    """LeaveOneGroupOut CV, grouped by subject.

    Returns per-fold (= per-epoch, pooled across folds) accuracy/kappa/balanced
    accuracy/macro-F1, AND the subject-level metric that actually matters here:
    every epoch from a held-out LOSO subject shares one true label, so the
    right per-subject prediction is the majority vote across that subject's
    epoch predictions, not a naive average of per-epoch accuracy.

    rf_params: optional dict of RandomForestClassifier kwargs, merged over the
    default (n_estimators=300, random_state=42, n_jobs=-1) -- used by
    run_loso_rf_configs() to compare regularized configs.
    """
    if channel_mask is not None:
        X = X[:, channel_mask]

    base_params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
    if rf_params:
        base_params.update(rf_params)

    logo = LeaveOneGroupOut()
    accs, kappas, bal_accs, f1s = [], [], [], []
    subject_correct, subject_true, subject_score = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = RandomForestClassifier(**base_params)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]

        accs.append(accuracy_score(y_true, y_pred))
        kappas.append(cohen_kappa_score(y_true, y_pred))
        bal_accs.append(balanced_accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # subject-level majority vote: all epochs in this fold are one subject
        majority_pred = np.bincount(y_pred).argmax()
        subject_correct.append(bool(majority_pred == y_true[0]))
        subject_true.append(int(y_true[0]))
        # subject-level "score" for AUC: mean predicted P(class=1) across
        # this subject's epochs (softer than the majority-vote hard label)
        y_proba = clf.predict_proba(X[test_idx])
        subject_score.append(float(y_proba[:, list(clf.classes_).index(1)].mean())
                              if 1 in clf.classes_ else 0.0)

    try:
        subject_level_auc = float(roc_auc_score(subject_true, subject_score))
    except ValueError:
        # can happen if every held-out subject ends up the same class in a
        # given run -- shouldn't happen across the full 45-subject LOSO here
        subject_level_auc = float("nan")

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_kappa": float(np.mean(kappas)),
        "mean_balanced_acc": float(np.mean(bal_accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "subject_level_acc": float(np.mean(subject_correct)),
        "subject_level_auc": subject_level_auc,
        "subject_correct": subject_correct,
        "per_fold_acc": accs,
        "n_folds": len(accs),
    }


def dummy_baseline(y, groups, strategy="stratified", seed=42):
    """LOSO with a DummyClassifier, as an explicit numeric chance-level
    reference to sit next to the real model's numbers -- instead of eyeballing
    "chance = 0.5". balanced_accuracy here should land at ~0.5 by construction
    regardless of strategy; useful mainly as a sanity check that our own
    balanced-accuracy computation is behaving as expected.
    """
    logo = LeaveOneGroupOut()
    accs, bal_accs = [], []
    X_dummy = np.zeros((len(y), 1))  # DummyClassifier ignores X entirely
    for train_idx, test_idx in logo.split(X_dummy, y, groups):
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(X_dummy[train_idx], y[train_idx])
        y_pred = clf.predict(X_dummy[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
    return {"mean_acc": float(np.mean(accs)), "mean_balanced_acc": float(np.mean(bal_accs))}


def run_loso_rf_configs(X, y, groups, configs=REGULARIZED_RF_CONFIGS, channel_mask=None):
    """Cheap, non-nested comparison of a few RF regularization settings (see
    REGULARIZED_RF_CONFIGS) on the same LOSO split -- directional check only,
    not a tuned/nested-CV search. Returns {config_name: run_loso() result}.
    """
    results = {}
    for name, params in configs.items():
        results[name] = run_loso(X, y, groups, channel_mask=channel_mask, rf_params=params)
    return results


def subject_level_features(X, y, groups):
    """Collapse each subject's many epochs into ONE feature vector (mean over
    epochs) and one label -- an alternative to epoch-level training +
    majority vote. Not a guaranteed fix (44 training subjects vs. thousands
    of epochs is its own small-sample regime), but cheap to try and a
    reasonable thing to compare against.

    Returns (n_subjects, n_features), (n_subjects,), (n_subjects,) -- the
    third array is the original subject ids, in the same order as the rows.
    """
    subjects = np.unique(groups)
    Xs, ys = [], []
    for s in subjects:
        mask = groups == s
        Xs.append(X[mask].mean(axis=0))
        ys.append(y[mask][0])
    return np.asarray(Xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), subjects


def run_loso_subject_level(Xs, ys, clf_factory, global_normalize=True):
    """LOSO where each 'sample' is already one subject (see
    subject_level_features): 44 train / 1 test per fold.

    global_normalize=True fits a single StandardScaler on the 44 training
    subjects each fold and applies it to the held-out one -- this is the
    statistically correct "global" normalization (distinct from the
    per-subject z-score that destroyed the label signal, since here the
    same shift/scale is applied to every subject rather than each subject
    being centered on itself). Only matters for scale-sensitive models like
    LDA/logistic regression -- irrelevant for Random Forest.
    """
    groups = np.arange(len(ys))  # each row is already its own subject
    logo = LeaveOneGroupOut()
    y_true, y_pred, y_score = [], [], []
    for train_idx, test_idx in logo.split(Xs, ys, groups):
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        if global_normalize:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train, ys[train_idx])
        pred = clf.predict(X_test)
        y_true.append(int(ys[test_idx][0]))
        y_pred.append(int(pred[0]))
        if hasattr(clf, "predict_proba"):
            classes = list(clf.classes_)
            y_score.append(float(clf.predict_proba(X_test)[0, classes.index(1)]) if 1 in classes else 0.0)
        else:
            y_score.append(float(clf.decision_function(X_test)[0]))

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    return {
        "subject_level_acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "subject_level_balanced_acc": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "subject_level_auc": auc,
        "n_subjects": len(y_true),
    }


def rank_channels_by_importance(X, y, groups, candidate_channels=FRONTAL_TEMPORAL):
    """Fit one RF on all candidate channels (LOSO-averaged importances) to get a
    ranking to drive the ablation curve, instead of brute-forcing all subset sizes.
    """
    mask = channel_feature_mask(candidate_channels)
    X_sub = X[:, mask]
    sub_names = [ch for ch in candidate_channels for _ in BANDS]

    logo = LeaveOneGroupOut()
    importances = np.zeros(X_sub.shape[1])
    n_folds = 0
    for train_idx, _ in logo.split(X_sub, y, groups):
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_sub[train_idx], y[train_idx])
        importances += clf.feature_importances_
        n_folds += 1
    importances /= n_folds

    # aggregate per-channel importance = sum over its bands
    per_channel = {}
    for ch, imp in zip(sub_names, importances):
        per_channel[ch] = per_channel.get(ch, 0) + imp

    ranked = sorted(per_channel.items(), key=lambda t: -t[1])
    return [ch for ch, _ in ranked]


def ablation_curve(X, y, groups, candidate_channels=FRONTAL_TEMPORAL):
    """Accuracy vs. number of channels, adding channels one at a time in order
    of importance (greedy forward selection), all evaluated with LOSO.
    """
    ranking = rank_channels_by_importance(X, y, groups, candidate_channels)

    results = []
    for k in range(1, len(ranking) + 1):
        subset = ranking[:k]
        mask = channel_feature_mask(subset)
        res = run_loso(X, y, groups, channel_mask=mask)
        results.append({"n_channels": k, "channels": subset, **res})
        print(f"n={k:2d}  channels={subset}  "
              f"acc={res['mean_acc']:.3f}±{res['std_acc']:.3f}  "
              f"bal_acc={res['mean_balanced_acc']:.3f}  macro_f1={res['mean_macro_f1']:.3f}  "
              f"subject_level_acc={res['subject_level_acc']:.3f}  kappa={res['mean_kappa']:.3f}")

    return pd.DataFrame(results)


def _print_full_result(label, res):
    print(f"[{label}] acc={res['mean_acc']:.3f}±{res['std_acc']:.3f}  "
          f"bal_acc={res['mean_balanced_acc']:.3f}  macro_f1={res['mean_macro_f1']:.3f}  "
          f"subject_level_acc={res['subject_level_acc']:.3f}  "
          f"subject_level_auc={res.get('subject_level_auc', float('nan')):.3f}  "
          f"kappa={res['mean_kappa']:.3f}  n_folds={res['n_folds']}  "
          f"(chance-level balanced_acc={CHANCE_LEVEL_BALANCED_ACC})")


if __name__ == "__main__":
    # Raw features (absolute log band power). CONFIRMED (both via teammate's
    # notebook and an independent rerun on the same .npz) this is the least-bad
    # of the three feature variants we've tried -- NOT "before", it's currently
    # the best available option. Per-subject normalization made things worse
    # (see methodology_note.md): the label is one value per whole session, so
    # per-subject z-scoring removes exactly the between-subject offset that
    # carries the label signal. Plain relative power (no per-subject norm)
    # was worse too. So the channel ablation below runs on RAW features.
    raw_path = DATA_DIR / "stew_features_two_class.npz"
    raw_data = np.load(raw_path, allow_pickle=True)
    Xr, yr, groupsr = raw_data["X"], raw_data["y"], raw_data["groups"]

    print("=== Chance-level reference (DummyClassifier, stratified LOSO) ===")
    dummy_res = dummy_baseline(yr, groupsr)
    print(f"acc={dummy_res['mean_acc']:.3f}  balanced_acc={dummy_res['mean_balanced_acc']:.3f}  "
          f"-- this (not the 0.556 majority-class ACCURACY) is the real chance-level "
          f"comparison point for balanced_acc / AUC below.")

    print("\n=== LOSO, all 14 channels, RAW (absolute band power), RF epoch-level + majority vote ===")
    raw_result = run_loso(Xr, yr, groupsr)
    _print_full_result("raw/absolute, RF", raw_result)

    # Alternative to epoch-level training + majority vote: collapse each
    # subject to one averaged feature vector (44 train / 1 test per LOSO
    # fold) and use models suited to small-n, e.g. LDA / logistic regression
    # instead of RF. Not guaranteed better -- just a cheap, fast experiment
    # worth comparing against the epoch-level result above.
    print("\n=== LOSO, subject-level aggregation (mean epoch per subject, 45 samples), "
          "global (train-fold) normalization ===")
    Xs, ys, _ = subject_level_features(Xr, yr, groupsr)
    for name, factory in [
        ("LDA", lambda: LinearDiscriminantAnalysis()),
        ("LogReg", lambda: LogisticRegression(max_iter=2000, C=1.0)),
    ]:
        res = run_loso_subject_level(Xs, ys, factory, global_normalize=True)
        print(f"[{name}] subject_level_acc={res['subject_level_acc']:.3f}  "
              f"balanced_acc={res['subject_level_balanced_acc']:.3f}  "
              f"auc={res['subject_level_auc']:.3f}  n={res['n_subjects']}  "
              f"(chance-level balanced_acc={CHANCE_LEVEL_BALANCED_ACC})")

    # Relative power / per-subject normalized features: kept here only for the
    # side-by-side comparison documented in methodology_note.md, NOT as input
    # to the ablation curve -- both variants underperformed raw in testing.
    norm_path = DATA_DIR / "stew_features_two_class_normalized.npz"
    if norm_path.exists():
        norm_data = np.load(norm_path, allow_pickle=True)
        X, y, groups = norm_data["X"], norm_data["y"], norm_data["groups"]

        print("\n=== LOSO, all 14 channels, relative power + per-subject z-score "
              "(comparison only -- confirmed worse, NOT used for ablation) ===")
        norm_result = run_loso(X, y, groups)
        _print_full_result("relative+per-subject-normalized", norm_result)

    # Combined absolute+relative power + engineered ratio/contrast features
    # (engagement index, frontal asymmetry) -- candidate improvement over raw
    # alone. Comparison only until it beats 0.578; not swapped in as default.
    eng_path = DATA_DIR / "stew_features_two_class_engineered.npz"
    if eng_path.exists():
        eng_data = np.load(eng_path, allow_pickle=True)
        Xe, ye, groupse = eng_data["X"], eng_data["y"], eng_data["groups"]
        print("\n=== LOSO, combined (absolute+relative) power + engineered features "
              "(engagement index, frontal asymmetry), RF epoch-level + majority vote ===")
        eng_result = run_loso(Xe, ye, groupse)
        _print_full_result("combined+engineered, RF", eng_result)

    # Quick regularization check (idea: 45 subjects / 70 features is a
    # small-n regime where an unconstrained RF can overfit). Non-nested --
    # directional signal only, run on RAW/absolute features for comparability
    # with the default RF result above.
    print("\n=== RF regularization comparison (RAW/absolute features, non-nested) ===")
    reg_results = run_loso_rf_configs(Xr, yr, groupsr)
    for name, res in reg_results.items():
        print(f"[{name}] subject_level_acc={res['subject_level_acc']:.3f}  "
              f"bal_acc={res['mean_balanced_acc']:.3f}  "
              f"subject_level_auc={res['subject_level_auc']:.3f}  "
              f"macro_f1={res['mean_macro_f1']:.3f}")

    print("\n=== Ablation (on RAW/absolute features): frontal/temporal channels only, LOSO ===")
    df = ablation_curve(Xr, yr, groupsr, candidate_channels=FRONTAL_TEMPORAL)
    df.to_csv(DATA_DIR / "ablation_curve.csv", index=False)
    print(f"\nSaved ablation curve to {DATA_DIR / 'ablation_curve.csv'}")
