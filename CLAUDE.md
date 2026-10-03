# MixLab

## Project goal
A Bayesian Marketing Mix Model (MMM) that estimates true channel contribution with adstock
(carryover) and saturation (diminishing returns), plus a budget optimizer and an AI layer that
explains results to a non-technical CMO.

## Data contract
- One row per week.
- `date`: week start date.
- `revenue`: the target.
- `spend_<channel>`: one spend column per channel (e.g. `spend_tv`, `spend_search`).
- Control variables: promo flags, holidays, price index, seasonality.
- All money is in INR.

## Coding rules
- Type hints on every function.
- Docstrings on every module and function.
- Small, pure functions.
- Configuration lives in `src/mixlab/config.py`; no magic numbers.
- Every module has tests.
- Set random seeds (use `config.RANDOM_SEED`).

## Workflow rule
- Before writing code, explain the plan in 5 bullets.
- After writing, run the tests and summarize what changed.

## Commands
- Fresh clone, end to end: `make data && make validate && make train && make app`
- Checks: `make lint`, `make test` (coverage must stay above 80%), `make evaluate`
- Deployment assets: `make deploy-assets` (slim models + `requirements.txt`); see `docs/DEPLOY.md`
- Install: `uv sync`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Generate data: `uv run python -m mixlab.data_gen --brand performance_heavy --seed 42`
- Re-run a notebook: `JUPYTER_PREFER_ENV_PATH=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb`
- Train model: `uv run python scripts/train.py --data data/synthetic/mmm_weekly.csv --config configs/default.yaml`
- Optimizer: `uv run python -m mixlab.optimizer` (writes `reports/optimizer_summary.json` and figures)
- AI brief: `uv run python -m mixlab.ai_explainer brief --tone cmo` (tones: cmo, analyst, founder)
- AI Q&A: `uv run python -m mixlab.ai_explainer ask "What happens if we pause TV?" --tone founder`
- AI examples: `uv run python -m mixlab.ai_explainer examples` (one brief per tone to `reports/ai_examples/`)
- AI facts (no API call): `uv run python -m mixlab.ai_explainer facts`
- Build demo brands: `uv run python scripts/build_demo.py` (fits three brands into `artifacts/`, about five minutes)
- Run the app: `uv run streamlit run app/main.py`
- Onboard real exports: `uv run python scripts/onboard.py --shopify orders.csv --meta meta.csv --google google.csv --out data/processed/mmm_weekly.csv [--anonymize]`
- Calibration demo: `uv run python scripts/calibrate_demo.py`
- Public dataset test: `uv run python scripts/public_dataset_test.py`

## Environment notes
- PyTensor compiles with Numba, not C (`mixlab/__init__.py` sets `PYTENSOR_FLAGS`); the C backend fails to link on this macOS toolchain. Import `mixlab` before `pymc`.
- Use `pymc_marketing.mmm.multidimensional.MMM`; the older `pymc_marketing.mmm.MMM` is deprecated.
- Insights: `uv run python -m mixlab.insights` (reads `models/`, writes `reports/insights_summary.json` and figures)
- PyMC-Marketing's `BudgetOptimizer` takes a per-week budget, not a period total; `optimizer.py` converts.
- The AI layer uses `claude-opus-5-5` via the Anthropic SDK; the key comes from `.env` (`ANTHROPIC_API_KEY`). Tests always mock the client; never call the live API from tests.
- The app never fits a model; it loads `artifacts/<brand>/`. App pages live in `app/views/` and are tested headlessly in `tests/test_app.py`.
