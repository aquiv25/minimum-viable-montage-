"""
Baseline sanity check: does the pipeline work at all, on all 14 channels?

This is deliberately the "easy" evaluation (random split, not subject-aware) --
its only job is to confirm the features + labels are sane before moving on to
the honest evaluation (LOSO, in stew_loso_ablation.py). Do not report this
number as the project's headline result: a random split lets epochs from the
same subject appear in both train and test, which inflates accuracy.

Defaults to the two_class dataset (not-overloaded vs overloaded, rating >= 7)
-- that's the project's actual target, not a rest-vs-task split, and it's the
one this dataset mirror actually supports (it only ships the SIMKAP task
recording per subject, no paired rest recording).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

from stew_pipeline import DATA_DIR


def run_baseline(npz_path=None):
    npz_path = npz_path or (DATA_DIR / "stew_features_two_class.npz")
    data = np.load(npz_path, allow_pickle=True)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"=== Baseline sanity check (random split, all 14 channels, {npz_path.name}) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Cohen's kappa: {cohen_kappa_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    # quick look at which channels/bands matter most, feeds into the ablation ranking
    importances = clf.feature_importances_
    names = list(data["feature_names"])
    ranked = sorted(zip(names, importances), key=lambda t: -t[1])
    print("Top 10 features by importance:")
    for name, imp in ranked[:10]:
        print(f"  {name}: {imp:.4f}")

    return clf


if __name__ == "__main__":
    run_baseline()
