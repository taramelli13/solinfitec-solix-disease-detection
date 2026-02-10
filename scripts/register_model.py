"""Register existing model checkpoints in MLflow Model Registry."""

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser(description="Register a checkpoint in MLflow")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model-name", required=True, help="Model registry name")
    parser.add_argument("--experiment", default="manual_registration")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"register_{args.model_name}"):
        mlflow.log_artifact(args.checkpoint)
        run_id = mlflow.active_run().info.run_id

    # Create registered model (ignore if already exists)
    try:
        client.create_registered_model(args.model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists from earlier run

    artifact_uri = f"runs:/{run_id}"
    client.create_model_version(
        name=args.model_name,
        source=artifact_uri,
        run_id=run_id,
    )

    print(f"Registered {args.model_name} from {args.checkpoint}")


if __name__ == "__main__":
    main()
