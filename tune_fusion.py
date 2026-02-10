"""Hyperparameter tuning for Fusion Model with Optuna + MLflow."""

import argparse
import sys
from pathlib import Path

import optuna
from optuna.integration.mlflow import MLflowCallback

sys.path.insert(0, str(Path(__file__).parent))
from train_fusion import train


def objective(trial):
    overrides = {
        "fusion_training.learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "temporal_model.d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "temporal_model.num_layers": trial.suggest_categorical("num_layers", [2, 4, 6]),
        "fusion.fused_dim": trial.suggest_categorical("fused_dim", [256, 512, 640]),
        "fusion_training.epochs": 15,
    }
    best_loss = train(
        config_path="configs/config.yaml",
        overrides=overrides,
    )
    # Optuna minimizes, and val_loss is already lower=better
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Tune Fusion Model hyperparameters")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name="fusion_hpo",
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
    )

    mlflow_cb = MLflowCallback(
        tracking_uri=args.tracking_uri,
        metric_name="best_val_loss",
    )

    study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflow_cb])

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best loss: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
