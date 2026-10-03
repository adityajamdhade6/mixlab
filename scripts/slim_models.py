"""Write a thinned copy of each demo brand's model, small enough to commit and deploy.

Example:
    uv run python scripts/slim_models.py

"""

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.model import MixLabModel


def main() -> None:
    """Slim every brand that has a full fitted model."""
    for name in config.BRAND_PRESETS:
        folder = config.ARTIFACTS_DIR / name
        if not (folder / config.MODEL_FILENAME).exists():
            print(f"{name}: no full model found, skipped")
            continue
        path = MixLabModel.load(folder).save_slim(folder)
        print(f"{name}: {path.stat().st_size / 1e6:.1f} MB -> {path.name}")


if __name__ == "__main__":
    main()
