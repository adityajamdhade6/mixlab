"""Convert platform exports into one weekly CSV that follows the MixLab data contract.

Example:
    uv run python scripts/onboard.py --shopify orders.csv --meta meta.csv --google google.csv \
        --out data/processed/mmm_weekly.csv --anonymize

With --anonymize, revenue and spend are scaled by a factor derived from the passphrase in the
MIXLAB_ANON_SECRET environment variable. The passphrase is never written anywhere.

"""

import argparse
import os
from pathlib import Path

import pandas as pd

from mixlab import config
from mixlab.onboarding import (
    OnboardingError,
    anonymize,
    build_contract_frame,
    factor_from_secret,
    map_google_ads,
    map_meta_ads,
    map_shopify_orders,
    read_export,
    write_template,
)
from mixlab.validate import validate


def main() -> None:
    """Map the given exports, validate the result and write it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shopify", type=Path, help="Shopify orders export (CSV).")
    parser.add_argument("--meta", type=Path, help="Meta Ads Manager export, daily (CSV).")
    parser.add_argument("--google", type=Path, help="Google Ads report, daily (CSV).")
    parser.add_argument("--out", type=Path, help="Where to write the weekly CSV.")
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--write-template", action="store_true", help="Only write the template.")
    args = parser.parse_args()

    if args.write_template:
        print(f"Wrote {write_template(config.TEMPLATES_DIR / config.DATA_TEMPLATE_FILENAME)}")
        return
    if not (args.shopify and args.out):
        parser.error("--shopify and --out are required (revenue comes from the Shopify export).")
    try:
        spend = []
        if args.meta:
            spend.append(map_meta_ads(pd.read_csv(args.meta)))
        if args.google:
            spend.append(map_google_ads(read_export(args.google, "Cost")))
        frame = build_contract_frame(map_shopify_orders(pd.read_csv(args.shopify)), spend)
        if args.anonymize:
            secret = os.environ.get(config.ANONYMIZE_SECRET_ENV, "")
            frame = anonymize(frame, factor_from_secret(secret))
    except OnboardingError as error:
        raise SystemExit(f"Could not map the exports: {error}") from error

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} weeks to {args.out}" + (" (anonymized)" if args.anonymize else ""))
    print(validate(frame).summary())


if __name__ == "__main__":
    main()
