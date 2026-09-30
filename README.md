# Product Sale Price Prediction — Pipeline

Five standalone scripts, one per pipeline stage. Each stage reads the
previous stage's output and writes its own results into a dedicated
`outputs/<stage>/` folder — nothing is shared in memory between scripts.

## Folder layout expected

```
project/
├── data/raw/BigBasket Products.csv     <- put your raw CSV here
├── pipeline/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── encoding.py
│   ├── model_building.py
│   └── model_evaluation.py
└── outputs/                             <- created automatically
    ├── preprocessing/
    ├── eda/
    ├── encoding/
    ├── model_building/
    └── model_evaluation/
```

## Run order

```bash
python pipeline/preprocessing.py
python pipeline/eda.py
python pipeline/encoding.py
python pipeline/model_building.py
python pipeline/model_evaluation.py
```

Every script also accepts `--input` / `--output-dir` (and `preprocessing.py`
takes `--input` for the raw CSV path) if you want a different layout — run
any script with `--help` to see its options.

## What each stage does

| Stage | Script | Reads | Writes to |
|---|---|---|---|
| 1 | `preprocessing.py` | raw CSV | `outputs/preprocessing/` — cleaned, aggregated, log-transformed product-level data + a text report |
| 2 | `eda.py` | preprocessing output | `outputs/eda/` — summary stats + distribution/correlation/category plots |
| 3 | `encoding.py` | preprocessing output | `outputs/encoding/` — one-hot encoded + scaled train/test CSVs, fitted encoder/scaler (`.pkl`), feature column list |
| 4 | `model_building.py` | encoding output | `outputs/model_building/` — fitted models (`.pkl`) + test-set predictions + training log |
| 5 | `model_evaluation.py` | model_building output | `outputs/model_evaluation/` — metrics table (RMSE/MAE/R2/MAPE), model comparison chart, actual-vs-predicted and residual plots, summary |

## Features used

- `rating` (mean per product)
- `market_log` = log1p(mean market price)
- `brandC_log` = log1p(mean brand popularity count)
- `product_count` = number of listings for that product
- `subcat_count` = mean sub-category popularity count
- one-hot encoded `category`

Target: `sale_price` (mean per product).

## Models trained

Linear Regression, Ridge, SGD Regressor, Decision Tree, Random Forest,
Gradient Boosting.
