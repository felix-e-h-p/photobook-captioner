"""Log the captioner to MLflow as a model and register a new version.

    uv run python scripts/log_model.py
    uv run mlflow models serve -m models:/photobook-captioner/latest --env-manager local -p 5001

Then caption a page over HTTP:

    curl -s localhost:5001/invocations -H 'Content-Type: application/json' \\
      -d '{"inputs": [{"images": ["data/skiing-holiday/Image_1.jpg"]}]}'
"""

from pathlib import Path

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "caption-model"
MODEL_NAME = "photobook-captioner"
EXAMPLE = [{"images": ["data/skiing-holiday/Image_1.jpg"]}]
MODEL_CODE = Path("src/captioner/serving.py")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="log-model"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MODEL_CODE,
            code_paths=["src/captioner"],
            pip_requirements=["pillow", "pycountry", "reverse-geocoder", "pydantic"],
            input_example=EXAMPLE,
            registered_model_name=MODEL_NAME,
        )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    print(f"logged {info.model_uri}")
    print("example:", loaded.predict(EXAMPLE))


if __name__ == "__main__":
    main()
