# MixLab developer guide

Commands, coding rules and environment notes for working on this repository.

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
- Refresh saved results without refitting: `uv run python scripts/build_demo.py --stage optimize` (or `backtest`)
- Architecture: `docs/architecture.md`
- Compare model variants: `uv run python scripts/compare_models.py` (LOO, holdout, recovery; about 30 minutes)
- Meridian benchmark (own environment): `uv run --isolated --python 3.11 --with google-meridian --with pandas python scripts/benchmark_meridian.py`
- Model decisions and their evidence: `docs/model_log.md`
- Re-estimate the optimizer's-curse haircut: `uv run python scripts/build_demo.py --stage curse` (six quick refits per brand)
- Optimizer benchmark on random brands: `uv run python scripts/optimizer_benchmark.py --brands 20`
- Geo panel (ten Indian regions): `uv run python -m mixlab.geo_data` (writes `geo_weekly.csv` and `geo_ground_truth.json`)
- Test-and-learn loop: `make experiments` or `uv run python scripts/experiment_loop.py` (one refit, about 5 minutes; `--refresh` recomputes the recommendations only)
- Geo demo for the Regions page: `make geo` or `uv run python scripts/build_geo_demo.py` (national + geo fits, about 30 minutes; `--stage analyze` recomputes from saved models)

## Model defaults (v2)
- `configs/default.yaml` and `configs/demo.yaml` select Hill saturation with an independent ROI prior. `ModelSettings()` with no file is still the v1 model (logistic, coefficient priors); tests rely on that.
- Hill models floor spend at one rupee (`config.HILL_MIN_SPEND`); exact zeros give an undefined gradient and thousands of divergences.
- After any model change: re-run `scripts/compare_models.py` and `scripts/evaluate_models.py`, keep the previous report in `reports/history/`, and do not merge if ground-truth recovery gets worse.

## Environment notes
- PyTensor compiles with Numba, not C (`mixlab/__init__.py` sets `PYTENSOR_FLAGS`); the C backend fails to link on this macOS toolchain. Import `mixlab` before `pymc`.
- Use `pymc_marketing.mmm.multidimensional.MMM`; the older `pymc_marketing.mmm.MMM` is deprecated.
- Insights: `uv run python -m mixlab.insights` (reads `models/`, writes `reports/insights_summary.json` and figures)
- PyMC-Marketing's `BudgetOptimizer` takes a per-week budget, not a period total; `optimizer.py` converts.
- The AI layer uses `claude-opus-5-5` via the Anthropic SDK; the key comes from `.env` (`ANTHROPIC_API_KEY`). Tests always mock the client; never call the live API from tests.
- The app never fits a model; it loads `artifacts/<brand>/`. App pages live in `app/views/` and are tested headlessly in `tests/test_app.py`.
- Geo data is long format: one row per (week, region) with a `geo` column. `geo_model.model_for(df)` picks `GeoMixLabModel` for a multi-region panel and the national `MixLabModel` otherwise. The geo model needs Hill saturation and an ROI prior (`configs/geo.yaml`).
- `extract_geo_draws` returns one `PosteriorDraws` per region with aligned draws, so every national tool (insights, optimizer maths) works per region and regional results can be summed draw by draw.
- The optimizer runs on posterior draws with SciPy (`optimizer.solve_allocation`); it does not need the PyMC model. `BudgetAllocator` (the library optimizer) is only a cross-check in tests.
