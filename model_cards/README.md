# Model cards

Catalog-ready metadata for each publishable weight set under `results/`.

## Layout

| Path | Role |
| --- | --- |
| `schema.json` | JSON Schema draft-2020-12 (`schema_version` **1.0.0**) |
| `cards/<family>__<run>.json` | One card per canonical checkpoint |
| `index.json` | Catalog index sorted by `primary_lpips` (lower better; nulls last) |

## Card granularity

One card = one **canonical** checkpoint for a train run (`latest_net_G.pth` or `ckpt_latest.pth` / `.pt`), not every epoch file. Patch evals (frozen test, train, organ holdouts, quant schemes, epoch ladders) attach under `evals[]`.

`primary_lpips` is denormalized for ranking and prefers frozen-test `lpips_convnext_tcga_mean`.

## Refresh

```bash
python scripts/build_model_cards.py
# optional (slow): python scripts/build_model_cards.py --hash-weights
```

The generator is idempotent. It rewrites `cards/` and `index.json` only.

## Extending

- New families: add a `FAMILY_TASK` entry in `scripts/build_model_cards.py`.
- Unknown future fields: put them in `extensions` (open object) or typed `training.hypers` / `metrics.extras`.
- New tasks: extend the `task` enum in `schema.json` (`i2i` | `encoder` | `autoencoder` | `other`).
