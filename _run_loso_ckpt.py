"""
Checkpointed LOSO runner -- fits/evaluates one fold at a time, saves progress
to a JSON file after each fold, and stops once a wall-clock time budget is
hit (so it can be resumed across several short invocations). Not a project
deliverable -- scratch tool for verifying real-data results within a
45s-per-call sandbox constraint.

Usage: python3 _run_loso_ckpt.py <npz_path> <ckpt_path> <n_estimators> [time_budget_sec] [max_depth] [min_samples_leaf]
max_depth: int or "none" (default "none")
min_samples_leaf: int (default 1)
"""
import sys, json, time, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut

npz_path, ckpt_path, n_estimators = sys.argv[1], sys.argv[2], int(sys.argv[3])
time_budget = float(sys.argv[4]) if len(sys.argv) > 4 else 35.0
max_depth_arg = sys.argv[5] if len(sys.argv) > 5 else "none"
max_depth = None if max_depth_arg.lower() == "none" else int(max_depth_arg)
min_samples_leaf = int(sys.argv[6]) if len(sys.argv) > 6 else 1

data = np.load(npz_path, allow_pickle=True)
X, y, groups = data["X"], data["y"], data["groups"]

logo = LeaveOneGroupOut()
folds = list(logo.split(X, y, groups))

if os.path.exists(ckpt_path):
    with open(ckpt_path) as f:
        ckpt = json.load(f)
else:
    ckpt = {"done": 0, "results": []}

start = time.time()
i = ckpt["done"]
while i < len(folds) and (time.time() - start) < time_budget:
    train_idx, test_idx = folds[i]
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    y_true = y[test_idx]
    y_proba = clf.predict_proba(X[test_idx])
    classes = list(clf.classes_)
    score = float(y_proba[:, classes.index(1)].mean()) if 1 in classes else 0.0
    majority_pred = int(np.bincount(y_pred).argmax())

    ckpt["results"].append({
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subject_true": int(y_true[0]),
        "subject_correct": bool(majority_pred == y_true[0]),
        "subject_score": score,
    })
    ckpt["done"] = i + 1
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)
    i += 1

print(f"progress: {ckpt['done']}/{len(folds)} folds done ({time.time()-start:.1f}s this call)")

if ckpt["done"] == len(folds):
    from sklearn.metrics import roc_auc_score
    res = ckpt["results"]
    subj_true = [r["subject_true"] for r in res]
    subj_score = [r["subject_score"] for r in res]
    subj_correct = [r["subject_correct"] for r in res]
    try:
        auc = roc_auc_score(subj_true, subj_score)
    except ValueError:
        auc = float("nan")
    print("=== FINAL ===")
    print(f"n_estimators={n_estimators} max_depth={max_depth} min_samples_leaf={min_samples_leaf}")
    print(f"mean_acc={np.mean([r['acc'] for r in res]):.4f}")
    print(f"mean_bal_acc={np.mean([r['bal_acc'] for r in res]):.4f}")
    print(f"mean_kappa={np.nanmean([r['kappa'] for r in res]):.4f}")
    print(f"mean_macro_f1={np.mean([r['f1'] for r in res]):.4f}")
    print(f"subject_level_acc={np.mean(subj_correct):.4f} ({sum(subj_correct)}/{len(subj_correct)})")
    print(f"subject_level_auc={auc:.4f}")
