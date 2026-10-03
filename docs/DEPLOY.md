# Deploying the dashboard to Streamlit Community Cloud

The app loads pre-fitted demo models; nothing is fitted on the server.

## One-time preparation (already done in this repo)
- `make deploy-assets` writes a slim copy of each model (`artifacts/<brand>/mixlab_mmm_slim.nc`,
  about 5 MB each, 1,000 draws) and `requirements.txt`. The full 100 MB models stay gitignored.
- The app falls back to the slim copy when the full model is absent.

## Steps
1. Commit and push the repository to GitHub, including `artifacts/` (JSON, CSV and the
   `*_slim.nc` files) and `requirements.txt`.
2. On share.streamlit.io choose **New app**, pick the repo and branch, and set the main file to
   `app/main.py`. Choose Python 3.11 under Advanced settings.
3. Under **Advanced settings > Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "your key"
   ```
   The app copies it into the environment at startup. Never commit the key. Without it the
   app still works: the Overview shows the standard summary and Ask MixLab explains that no
   key is set.
4. Deploy. The first load compiles the model with Numba and takes a minute or two.

## Things to know
- The slim models have fewer draws than the saved summaries, so numbers computed live
  (optimizer, scenarios, response curves) can differ from the Overview in the second decimal.
- Community Cloud gives about 1 GB of memory. One brand's model loaded at a time fits; loading
  all three in one session is close to the limit.
- A public app with an API key attached lets visitors spend your credits through Ask MixLab.
  Leave the secret out for a public demo, or set a spend limit on the key.
