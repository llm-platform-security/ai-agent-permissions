# Data Documentation

Anonymized data from our user study on data-sharing preferences in AI agents.

## Files

### 1. `data_types.csv`
Catalog of 77 data types across categories: Personal, Health, Finance, Location, Smart Home, etc.

### 2. `user_study.json`
Anonymized responses from 203 participants.

**Structure:**
```json
{
  "uuid": "P001",
  "education": "Master's",
  "aiFamiliarity": "4",
  "aiTrust": "3",
  "privacyImportance": "4",
  "concern_domain": ["Finance", "Health & Fitness"],
  "task1": [...],  // App selection
  "task2": [...],  // Data selection
  "task3": [...]   // Permission preferences
}
```

### 3. `processed_dataset.json`
Processed dataset with 181 filtered participants (≥5 "always share" or "never share" decisions).

**Structure:**
```json
{
  "P001": {
    "bio": {...},
    "ai_experience": {...},
    "training": [...]
  }
}
```

## Data Schema

### queries.json (in root directory)

65 study scenarios with ground truth labels.

**Key fields:**
- `query`: User request text
- `domain`: Health, Finance, Shopping, Travel, Work, Entertainment, Social, Smart Home
- `datatype`: List of necessary data types
- `tool1`, `tool2`: Required applications
- `id`: Query identifier (0-64)

### Permission Levels

- `"Yes, always share"` - Grant automatically
- `"Yes, but ask me first"` - Ask before sharing
- `"No, but ask me first"` - Deny but allow override
- `"No, never share"` - Always deny

## User Study

**Participants:** 203 (via Prolific)

**Tasks:**
1. Select apps needed for each query
2. Identify necessary data types
3. Set data-sharing preferences

**Domains:** Health & Fitness, Finance, Shopping, Travel, Work, Entertainment, Social, Smart Home

See `website.pdf` in root for interface screenshots.

## Anonymization

- All Prolific IDs replaced with P001-P203
- No personally identifiable information
