# Minimum Viable Montage — NSRI Summer Research Hackathon 2026

How few EEG channels does a consumer headband actually need to reliably detect
cognitive overload in a new user? See `project_plan.md` (team plan, in Russian)
and `methodology_note.md` (methodology, in English) for the full framing.

## Start here

**`notebooks/minimum_viable_montage_analysis.ipynb`** is the consolidated,
polished analysis — data loading, baseline model, normalization tests, feature
engineering, significance testing, and the channel-ablation curve, all in one
place with commentary. This is the notebook to share with teammates or the
professor.

## Repository layout

```
stew_pipeline.py          Feature extraction: raw .mat -> band-power features (build_dataset, load_raw, CHANNELS, BANDS)
stew_loso_ablation.py     LOSO evaluation + channel ablation curve (run_loso, ablation_curve, rank_channels_by_importance)
stew_baseline.py          Quick sanity-check only (random split, NOT the project's headline result — see file docstring)

notebooks/
  minimum_viable_montage_analysis.ipynb   Final consolidated analysis notebook (see above)
  build_notebook.py                       Script that generates the notebook above, for transparency/reproducibility

data/stew/
  dataset.mat, rating.mat, class_012.mat,
  three_class_one_hot.mat                 Raw STEW dataset (45 subjects, 14-channel Emotiv EPOC, SIMKAP task)
  stew_features_two_class*.npz            Cached extracted feature sets (absolute / normalized / engineered / extended)
  ckpt_*.json                             Cached LOSO results per model config (checkpointed fold-by-fold; see below)
  channel_ranking.json, ablation_curve.json/.png   Channel-importance ranking + accuracy-vs-channel-count curve
  _run_loso_ckpt.py, _run_loso_ablation_ckpt.py    Checkpoint-and-resume LOSO runner scripts (used to produce the ckpt_*.json files)

reference_materials/
  EEG data summary.pdf                    STEW dataset documentation
```

### Which `ckpt_*.json` is which

| File | Config | Result |
|---|---|---|
| `ckpt_raw_n100.json` | RF default (unregularized), 70 absolute-power features | 26/45 (0.578) |
| `ckpt_normalized_n100.json` | RF default, relative power + per-subject z-score | 25/45 (0.556) — normalization hurts |
| `ckpt_tuned_d6_l5_n100/n200.json` | RF `max_depth=6, min_samples_leaf=5`, 70 features | 28/45 (0.622) |
| `ckpt_teammate_exact_n200.json` / `ckpt_88feat_d6_l5_n*.json` | Same tuned RF, 88 features (+ alpha/beta ratio + frontal asymmetry) | 29/45 (0.644) — matches teammate's claim exactly, but not significantly different from the 70-feature version (McNemar p=1.0) |
| `ckpt_k1.json` … `ckpt_k9.json` | Tuned RF, cumulative channel subsets (ablation curve) | 28–29/45 across the board — flat curve |

## Reproducing

```
pip install -r requirements.txt
```

Then open `notebooks/minimum_viable_montage_analysis.ipynb` and run all cells
(run from within the `notebooks/` directory so the relative paths resolve).
Expensive LOSO runs (several minutes each) are loaded from the cached
`ckpt_*.json` files rather than recomputed live; the scripts used to produce
those caches are included in `data/stew/` if you want to regenerate them from
scratch.

## Key findings (see the notebook for full detail)

1. Absolute band power beats every normalization variant tried.
2. Limiting Random Forest tree depth is the one robust, real improvement over the unregularized default.
3. Adding engineered features (alpha/beta ratio, frontal asymmetry) reaches 64.4%, at the edge of significance (p<0.05 at n=45) — but a McNemar test shows this is a single-subject-flip effect, not distinguishable from noise.
4. Channel count barely matters: a 2-channel frontal montage (F3, AF3) performs statistically indistinguishably from the full 10- or 14-channel montage — the strongest, best-supported basis for "Minimum Viable Montage."
5. n=45 gives limited statistical power; every result here is directional, not a strongly confirmed finding.
