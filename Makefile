.DEFAULT_GOAL := help
.PHONY: help install test lint format check evaluate evaluate-llm sweep review score model mlflow serve docker docker-run clean outputs figures

RUN ?=
IMAGE := photobook-captioner

help:
	@grep -E '^[a-z-]+:' $(MAKEFILE_LIST) | cut -d: -f1 | sed 's/^/  make /'

install:
	uv sync
	uv run pre-commit install

test:
	uv run pytest --cov=captioner --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint test

evaluate:
	uv run python scripts/evaluate.py

evaluate-llm:
	uv run python scripts/evaluate.py writer=llm judge.enabled=true clip.enabled=true

review:
	uv run python scripts/review.py make $(RUN)

score:
	uv run python scripts/review.py score $(RUN)

model:
	uv run python scripts/log_model.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:
	uv run uvicorn captioner.api:app --reload --port 8001

docker:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8001:8000 -v "$(CURDIR)/data:/photos:ro" $(IMAGE)

sweep:
	uv run python scripts/evaluate.py -m writer=template,llm use_context=true,false

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

outputs:
	uv run python scripts/examples.py
	uv run python scripts/examples.py --llm

figures:
	uv run python scripts/figures.py
