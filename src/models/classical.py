"""
Classical machine learning models for biomedical classification.
Includes logistic regression, decision trees, random forests, SVM, and ensemble methods.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _probabilistic_svc(**svc_params):
    """SVC with `predict_proba` via Platt scaling (replaces the deprecated `SVC(probability=True)`)."""
    return CalibratedClassifierCV(SVC(**svc_params), ensemble=False)

# name -> (factory taking random_state, description)
_MODELS = {
    "logistic_regression": (
        lambda rs: LogisticRegression(max_iter=1000, random_state=rs, solver='lbfgs'),
        "Linear model using logistic function",
    ),
    "logistic_regression_saga": (
        lambda rs: LogisticRegression(max_iter=2000, random_state=rs, solver='saga'),
        "Logistic regression with the SAGA solver (suited to large datasets)",
    ),
    "decision_tree": (
        lambda rs: DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=rs,
        ),
        "Tree-based model with interpretable decision boundaries",
    ),
    "random_forest": (
        lambda rs: RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            random_state=rs, n_jobs=-1,
        ),
        "Ensemble of decision trees",
    ),
    "gradient_boosting": (
        lambda rs: GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, min_samples_split=5,
            min_samples_leaf=2, random_state=rs,
        ),
        "Boosting ensemble method",
    ),
    "svm": (
        lambda rs: _probabilistic_svc(kernel='rbf', C=1.0, gamma='scale'),
        "Support Vector Machine with RBF kernel",
    ),
    "svm_linear": (
        lambda rs: _probabilistic_svc(kernel='linear', C=1.0),
        "Support Vector Machine with linear kernel",
    ),
    "svm_poly": (
        lambda rs: _probabilistic_svc(kernel='poly', degree=3, C=1.0),
        "Support Vector Machine with polynomial kernel",
    ),
    "knn": (
        lambda rs: KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=-1),
        "K-Nearest Neighbors with uniform weights",
    ),
    "knn_weighted": (
        lambda rs: KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        "K-Nearest Neighbors with distance-based weights",
    ),
    "naive_bayes": (
        lambda rs: GaussianNB(),
        "Probabilistic classifier based on Bayes theorem",
    ),
    "adaboost": (
        lambda rs: AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=rs),
        "Adaptive boosting ensemble",
    ),
}


def make_classical_model(name="random_forest", random_state=42):
    """
    Create a classical machine learning model by name.

    Args:
        name: Model name (see `get_model_list()`)
        random_state: Random seed for reproducibility

    Returns:
        Sklearn model instance
    """
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f"Unsupported classical model: {name}. Choose from {get_model_list()}")
    factory, _ = _MODELS[name]
    return factory(random_state)


def make_ensemble_model(random_state=42):
    """
    Create a soft-voting ensemble combining multiple classifiers.

    Args:
        random_state: Random seed

    Returns:
        Voting classifier ensemble
    """
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("svm", _probabilistic_svc(kernel='rbf')),
        ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
    ]
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)


def make_stacking_model(random_state=42):
    """
    Create a stacking classifier with base learners and meta-learner.

    Args:
        random_state: Random seed

    Returns:
        Stacking classifier
    """
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=random_state)),
        ("svm", SVC(kernel='rbf')),
    ]
    meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
    return StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)


def make_clustering_model(n_clusters=3, random_state=42):
    """
    Create a K-means clustering model.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        KMeans model
    """
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)


def get_model_list():
    """Get list of available classical models."""
    return list(_MODELS)


def get_model_info(name):
    """Get a short description of a specific model."""
    entry = _MODELS.get(name)
    return entry[1] if entry else "Unknown model"
