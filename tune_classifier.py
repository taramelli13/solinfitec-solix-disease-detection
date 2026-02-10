"""Hyperparameter tuning for Swin Classifier with Optuna + MLflow."""

import argparse
import sys
from pathlib import Path

import optuna
from optuna.integration.mlflow import MLflowCallback

sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import train


def objective(trial):
    overrides = {
        "training.learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 0.01, 0.1, log=True),
        "training.focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
        "data.batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "training.epochs": 15,
    }
    best_f1 = train(config_path="configs/config.yaml", overrides=overrides)
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Tune Swin Classifier hyperparameters")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name="swin_hpo",
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    mlflow_cb = MLflowCallback(
        tracking_uri=args.tracking_uri,
        metric_name="best_val_f1",
    )

    study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflow_cb])

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
