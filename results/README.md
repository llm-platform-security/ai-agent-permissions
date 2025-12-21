# Results Directory

Output files from the three permission prediction methods.

## Files

### CF Only (Collaborative Filtering)
- `cf_only_predictions.json` - Test predictions with ground truth
- `cf_only_metrics.json` - Performance metrics
- `cf_scores.csv` - CF scores for all user-query pairs (required by IC+CF)

### IC Only (In-Context Learning)
- `ic_only_predictions.json` - LLM predictions with ground truth
- `ic_only_metrics.json` - Performance metrics

### IC+CF (Hybrid Approach)
- `ic_cf_predictions.json` - LLM predictions with CF context
- `ic_cf_metrics.json` - Performance metrics

## Metrics Format

Each metrics file contains:
- `method` - Method name
- `accuracy`, `precision`, `recall`, `f1` - Performance metrics
- `fpr`, `fnr` - False positive/negative rates
- `n_predictions` - Number of predictions

## Usage

From the `src/` directory:

```bash
# 1. Generate CF-only results (required first for IC+CF)
python permission_cf_only.py

# 2. Generate IC-only results
python permission_ic_only.py

# 3. Generate IC+CF results (requires cf_scores.csv)
python permission_ic_cf.py
```

All results are automatically saved to this directory.
