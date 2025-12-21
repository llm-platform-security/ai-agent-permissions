# Source Code

Core implementation of the three experimental approaches: CF-only, IC-only, and IC+CF hybrid.

## Files

- `permission_cf_only.py` - Collaborative filtering baseline (LightGCN)
- `permission_ic_only.py` - In-context learning baseline
- `permission_ic_cf.py` - IC+CF hybrid (main contribution)
- `evaluation_utils.py` - Shared evaluation utilities

## Usage

Run scripts in order from the `src/` directory:

```bash
# 1. CF only - Generates CF scores (no API key required)
python permission_cf_only.py

# 2. IC only - Requires OpenAI API key
python permission_ic_only.py

# 3. IC+CF - Requires OpenAI API key + CF scores from step 1
python permission_ic_cf.py
```

**Requirements:**
- IC-only and IC+CF require `OPENAI_API_KEY` in `.env` file
- All dependencies: `pip install -r ../requirements.txt`

## Outputs

All results automatically saved to `../results/`:
- `{method}_predictions.json` - Test predictions
- `{method}_metrics.json` - Performance metrics
- `cf_scores.csv` - CF scores (required by IC+CF)

See `../results/README.md` for details.
