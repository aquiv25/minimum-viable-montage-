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

CHANCE_LEVEL_BALANCED_ACC = 0.5

REGULARIZED_RF_CONFIGS = {
    "default (unregularized)": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    "max_depth=5": {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 1},
    "max_depth=10": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 1},
    "min_samples_leaf=10": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10},
    "max_depth=10, min_samples_leaf=5": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
}

# The config actually used for the project's headline result (64.4%, see
# README / notebook). Passed explicitly into ablation_curve() by default so
# the channel-ablation search uses the SAME tuned settings as the headline
# result, instead of silently falling back to an unregularized RF.
TUNED_RF_PARAMS = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5}


def channel_feature_mask(selected_channels, all_channels=CHANNELS, bands=BANDS):
    n_bands = len(bands)
    mask = np.zeros(len(all_channels) * n_bands, dtype=bool)
    for ch in selected_channels:
        idx = all_channels.index(ch)
        mask[idx * n_bands:(idx + 1) * n_bands] = True
    return mask


def run_loso(X, y, groups, channel_mask=None, rf_params=None):
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

        majority_pred = np.bincount(y_pred).argmax()
        subject_correct.append(bool(majority_pred == y_true[0]))
        subject_true.append(int(y_true[0]))
        y_proba = clf.predict_proba(X[test_idx])
        subject_score.append(float(y_proba[:, list(clf.classes_).index(1)].mean())
                              if 1 in clf.classes_ else 0.0)

    try:
        subject_level_auc = float(roc_auc_score(subject_true, subject_score))
    except ValueError:
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
    logo = LeaveOneGroupOut()
    accs, bal_accs = [], []
    X_dummy = np.zeros((len(y), 1))
    for train_idx, test_idx in logo.split(X_dummy, y, groups):
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(X_dummy[train_idx], y[train_idx])
        y_pred = clf.predict(X_dummy[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
    return {"mean_acc": float(np.mean(accs)), "mean_balanced_acc": float(np.mean(bal_accs))}


def run_loso_rf_configs(X, y, groups, configs=REGULARIZED_RF_CONFIGS, channel_mask=None):
    results = {}
    for name, params in configs.items():
        results[name] = run_loso(X, y, groups, channel_mask=channel_mask, rf_params=params)
    return results


def subject_level_features(X, y, groups):
    subjects = np.unique(groups)
    Xs, ys = [], []
    for s in subjects:
        mask = groups == s
        Xs.append(X[mask].mean(axis=0))
        ys.append(y[mask][0])
    return np.asarray(Xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), subjects


def run_loso_subject_level(Xs, ys, clf_factory, global_normalize=True):
    groups = np.arange(len(ys))
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


def rank_channels_by_importance(X, y, groups, candidate_channels=FRONTAL_TEMPORAL, rf_params=None):
    """rf_params: pass the SAME rf_params used for the headline result (e.g.
    TUNED_RF_PARAMS) so the importance ranking reflects the tuned model, not
    an unregularized default.
    """
    mask = channel_feature_mask(candidate_channels)
    X_sub = X[:, mask]
    sub_names = [ch for ch in candidate_channels for _ in BANDS]

    base_params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
    if rf_params:
        base_params.update(rf_params)

    logo = LeaveOneGroupOut()
    importances = np.zeros(X_sub.shape[1])
    n_folds = 0
    for train_idx, _ in logo.split(X_sub, y, groups):
        clf = RandomForestClassifier(**base_params)
        clf.fit(X_sub[train_idx], y[train_idx])
        importances += clf.feature_importances_
        n_folds += 1
    importances /= n_folds

    per_channel = {}
    for ch, imp in zip(sub_names, importances):
        per_channel[ch] = per_channel.get(ch, 0) + imp

    ranked = sorted(per_channel.items(), key=lambda t: -t[1])
    return [ch for ch, _ in ranked]


def ablation_curve(X, y, groups, candidate_channels=FRONTAL_TEMPORAL, rf_params=TUNED_RF_PARAMS):
    """rf_params defaults to TUNED_RF_PARAMS -- the exact config the
    project's 64.4% headline result and the committed ckpt_k*.json files
    were generated with. Pass rf_params=None to reproduce the old
    (unregularized) behavior instead. This fixes the earlier bug where
    ablation_curve() silently ignored rf_params entirely.
    """
    ranking = rank_channels_by_importance(X, y, groups, candidate_channels, rf_params=rf_params)

    results = []
    for k in range(1, len(ranking) + 1):
        subset = ranking[:k]
        mask = channel_feature_mask(subset)
        res = run_loso(X, y, groups, channel_mask=mask, rf_params=rf_params)
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
    raw_path = DATA_DIR / "stew_features_two_class.npz"
    raw_data = np.load(raw_path, allow_pickle=True)
    Xr, yr, groupsr = raw_data["X"], raw_data["y"], raw_data["groups"]

    print("=== Chance-level reference (DummyClassifier, stratified LOSO) ===")
    dummy_res = dummy_baseline(yr, groupsr)
    print(f"acc={dummy_res['mean_acc']:.3f}  balanced_acc={dummy_res['mean_balanced_acc']:.3f}")

    print("\n=== LOSO, all 14 channels, RAW (absolute band power), RF epoch-level + majority vote ===")
    raw_result = run_loso(Xr, yr, groupsr)
    _print_full_result("raw/absolute, RF", raw_result)

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
              f"auc={res['subject_level_auc']:.3f}  n={res['n_subjects']}")

    norm_path = DATA_DIR / "stew_features_two_class_normalized.npz"
    if norm_path.exists():
        norm_data = np.load(norm_path, allow_pickle=True)
        X, y, groups = norm_data["X"], norm_data["y"], norm_data["groups"]
        print("\n=== LOSO, all 14 channels, relative power + per-subject z-score "
              "(comparison only) ===")
        norm_result = run_loso(X, y, groups)
        _print_full_result("relative+per-subject-normalized", norm_result)

    eng_path = DATA_DIR / "stew_features_two_class_engineered.npz"
    if eng_path.exists():
        eng_data = np.load(eng_path, allow_pickle=True)
        Xe, ye, groupse = eng_data["X"], eng_data["y"], eng_data["groups"]
        print("\n=== LOSO, combined (absolute+relative) power + engineered features, "
              "RF epoch-level + majority vote ===")
        eng_result = run_loso(Xe, ye, groupse)
        _print_full_result("combined+engineered, RF", eng_result)

    print("\n=== RF regularization comparison (RAW/absolute features, non-nested) ===")
    reg_results = run_loso_rf_configs(Xr, yr, groupsr)
    for name, res in reg_results.items():
        print(f"[{name}] subject_level_acc={res['subject_level_acc']:.3f}  "
              f"bal_acc={res['mean_balanced_acc']:.3f}  "
              f"subject_level_auc={res['subject_level_auc']:.3f}  "
              f"macro_f1={res['mean_macro_f1']:.3f}")

    # FIX: now runs with rf_params=TUNED_RF_PARAMS by default (see
    # ablation_curve()'s signature) instead of silently using an
    # unregularized RF -- this matches the tuned config the headline result
    # and committed ckpt_k*.json files used.
    print("\n=== Ablation (on RAW/absolute features, TUNED RF): "
          "frontal/temporal channels only, LOSO ===")
    df = ablation_curve(Xr, yr, groupsr, candidate_channels=FRONTAL_TEMPORAL)
    df.to_csv(DATA_DIR / "ablation_curve.csv", index=False)
    print(f"\nSaved ablation curve to {DATA_DIR / 'ablation_curve.csv'}")
