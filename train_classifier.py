import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump


def train_classifier(X: np.ndarray, y: np.ndarray, model_path: str):
    """Train an SVM-RBF classifier with grid search and save the best model."""
    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    }
    svc = SVC(probability=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid = GridSearchCV(svc, param_grid, cv=cv, n_jobs=-1)
    grid.fit(X, y)

    best = grid.best_estimator_
    dump(best, model_path)

    preds = best.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="binary")
    rec = recall_score(y, preds, average="binary")
    f1 = f1_score(y, preds, average="binary")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    # use 22 features to mimic the pipeline output size
    X, y = make_classification(n_samples=200, n_features=22, random_state=42)
    train_classifier(X, y, "svm_model.joblib")
