# MixLab: a fresh clone runs end to end with four commands (uv installs everything on first use):
#   make data      generate the three synthetic demo brands
#   make validate  check each dataset against the data contract
#   make train     fit each brand and save insights and optimizer results (about 5 minutes)
#   make app       open the dashboard

BRANDS_DIR := artifacts

.PHONY: data validate train app test lint format evaluate calibrate template deploy-assets clean

data:
	uv run python scripts/build_demo.py --stage data

validate:
	uv run python -m mixlab.validate $(BRANDS_DIR)/*/mmm_weekly.csv

train:
	uv run python scripts/build_demo.py --stage train

app:
	uv run streamlit run app/main.py

test:
	uv run pytest --cov=src/mixlab --cov-report=term-missing --cov-fail-under=80

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix
	uv run ruff format .

evaluate:
	uv run python scripts/evaluate_models.py

calibrate:
	uv run python scripts/calibrate_demo.py

template:
	uv run python scripts/onboard.py --write-template

deploy-assets:
	uv run python scripts/slim_models.py
	uv export --no-dev --no-hashes -o requirements.txt

clean:
	rm -rf .cache .pytest_cache .ruff_cache .coverage
