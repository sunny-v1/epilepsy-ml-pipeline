"""Smoke-test the project environment and run a minimal end-to-end pipeline.

Checks:
- running on Python 3.15+ and required packages import
- breast cancer dataset -> feature engineering -> preprocessing -> random forest -> metrics
- a small PyTorch MLP trains for a few epochs
- the epileptic seizure loader works offline (a missing local CSV is OK)

Exit codes:
- 0: success
- 1: wrong Python version
- 2: import/module error
- 3: pipeline runtime error
"""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

print("Python:", sys.version.split()[0])
if sys.version_info < (3, 15):
    print("Python 3.15+ is required.")
    sys.exit(1)

try:
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "torch", "pywt"):
        print(f"{pkg}:", importlib.import_module(pkg).__version__)

    from src.data.loader import load_epileptic_seizure, load_public_dataset
    from src.evaluation.metrics import binary_classification_summary
    from src.features.engineer import enrich_features
    from src.models.classical import make_classical_model
    from src.models.deep import SimpleMLP
    from src.preprocessing.pipeline import build_preprocessing_pipeline
    from src.training.train import predict_torch, split_data, train_torch
    from src.utils.seed import set_seed
except ImportError as e:
    print("Import failed:", e)
    sys.exit(2)

try:
    set_seed(42)

    df, meta = load_public_dataset("breast_cancer")
    features = enrich_features(df.drop(columns=["target"]))
    df = features.assign(target=df["target"])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preproc = build_preprocessing_pipeline(k_best=20, add_poly=False)
    X_train_p = preproc.fit_transform(X_train, y_train)
    X_val_p = preproc.transform(X_val)
    X_test_p = preproc.transform(X_test)

    model = make_classical_model("random_forest").fit(X_train_p, y_train)
    summary = binary_classification_summary(
        y_test, model.predict(X_test_p), model.predict_proba(X_test_p)
    )
    print(f"Random forest: accuracy={summary['accuracy']:.3f} roc_auc={summary['roc_auc']:.3f}")

    mlp = SimpleMLP(input_dim=X_train_p.shape[1], n_classes=meta["n_classes"])
    mlp, _ = train_torch(mlp, X_train_p, y_train, X_val_p, y_val, epochs=5, verbose=False)
    pred, _ = predict_torch(mlp, X_test_p)
    print(f"MLP: accuracy={(pred == y_test.to_numpy()).mean():.3f}")

    try:
        epi_df, _ = load_epileptic_seizure(openml_lookup=False)
        print("Epileptic seizure CSV loaded:", epi_df.shape)
    except FileNotFoundError:
        print("Epileptic seizure CSV not found at data/epileptic_seizure.csv (OK).")
except Exception as e:
    print("Pipeline raised unexpected error:", repr(e))
    sys.exit(3)

print("Smoke test passed")
