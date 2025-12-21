# Towards Automating Data Access Permissions in AI Agents

[![IEEE S&P 2026](https://img.shields.io/badge/IEEE%20S%26P-2026-blue.svg)](https://doi.ieeecomputersociety.org/10.1109/SP63933.2026.00018)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17959-b31b1b.svg)](https://arxiv.org/abs/2511.17959)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

## Overview

This repository contains the data, source code, and materials for our research on automating data access permissions in AI agents. We take a multifaceted approach: (i) conducting a vignette-based user study to understand user permission preferences, and (ii) developing a permission prediction system that combines LLM in-context learning with collaborative filtering.

### Key Contributions

1. **Automated Permission Framework**: We propose automating data access permissions in AI agents through a permission assistant that observes user history and makes automatic decisions.

2. **Vignette-Based User Study**: We develop a bespoke vignette-based user study to understand factors influencing users' data-sharing permission decisions in AI agents.

3. **Hybrid Permission Inference System**: We develop a hybrid permission inference framework combining in-context learning and collaborative filtering to predict user preferences.

## Repository Structure

```
ai-agent-permissions/
├── data/                       # User study data and documentation
│   ├── README.md               # Data documentation
│   ├── data_types.csv          # Data types catalog (77 data types)
│   ├── user_study.json         # Anonymized user responses (203 participants)
│   └── processed_dataset.json  # Processed dataset (181 filtered participants)
├── src/                        # Source code implementing three experimental approaches
│   ├── README.md               # Source code documentation
│   ├── permission_cf_only.py   # CF only: LightGCN collaborative filtering
│   ├── permission_ic_only.py   # IC only: In-context learning baseline
│   ├── permission_ic_cf.py     # IC+CF: Hybrid approach (main contribution)
│   └── evaluation_utils.py     # Shared evaluation utilities
├── results/                    # Output directory for all experimental results
│   └── README.md               # Output files documentation
├── queries.json                # 65 study scenarios with ground truth
├── website.pdf                 # User study website screenshots
├── requirements.txt            # Python dependencies
├── LICENSE                     # CC BY 4.0 license
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (for IC-only and IC+CF methods)

### Installation

```bash
# Clone the repository
git clone https://github.com/llm-platform-security/ai-agent-permissions.git
cd ai-agent-permissions

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create `.env` file from template and configure your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
# Required for IC-only and IC+CF methods
OPENAI_API_KEY=your-api-key-here

# Optional: Configure model (defaults to o4-mini)
OPENAI_MODEL=o4-mini              # Default (more advanced)
# OPENAI_MODEL=o3-mini-2025-01-31  # Used in the paper
```

## Running the Code

The source code implements the three experimental approaches. All outputs are saved to `results/` directory.

```bash
cd src

# 1. CF only: Collaborative filtering with LightGCN (no API key required)
python permission_cf_only.py

# 2. IC only: In-context learning without collaborative filtering
python permission_ic_only.py

# 3. IC+CF: In-context learning with collaborative filtering (hybrid approach)
# Requires cf_scores.csv from step 1
python permission_ic_cf.py
```

**Notes:**
- Run `permission_cf_only.py` first to generate `cf_scores.csv` required by `permission_ic_cf.py`
- IC-only and IC+CF require `OPENAI_API_KEY` in `.env` file
- All results saved to `results/` directory

**Documentation:**
- See [`src/README.md`](src/README.md) for source code documentation
- See [`results/README.md`](results/README.md) for output files documentation

## Data

All data has been anonymized to protect participant privacy:

- **User Study Data**: Prolific IDs replaced with anonymous participant IDs (P001-P203)
- **Queries**: 65 scenarios spanning 8 domains (Health & Fitness, Finance, Shopping, Travel, etc.)
- **Responses**: Participant choices for app selection, data selection, and permission preferences

**Documentation:**
- See [`data/README.md`](data/README.md) for data documentation

**User Study Metadata:**
- [`queries.json`](queries.json) - Study scenarios with ground truth labels
- [`website.pdf`](website.pdf) - User study interface screenshots

## Research Team

[Yuhao Wu](https://yuhao-w.github.io) (Washington University in St. Louis)  
[Ke Yang](https://www.linkedin.com/in/ke-yang-b46432294/) (University of California, Irvine)  
[Franziska Roesner](https://www.franziroesner.com/) (University of Washington)  
[Tadayoshi Kohno](https://gufaculty360.georgetown.edu/s/contact/003UH00000ZmP3HYAV/yoshi-kohno) (Georgetown University)  
[Ning Zhang](https://cybersecurity.seas.wustl.edu/) (Washington University in St. Louis)  
[Umar Iqbal](https://umariqbal.com) (Washington University in St. Louis)  

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{wu2026automating,
  title={{Towards Automating Data Access Permissions in AI Agents}},
  author={Wu, Yuhao and Yang, Ke and Roesner, Franziska and Kohno, Tadayoshi and Zhang, Ning and Iqbal, Umar},
  booktitle={2026 IEEE Symposium on Security and Privacy (SP)},
  pages={336--354},
  year={2026},
  organization={IEEE},
  doi={10.1109/SP63933.2026.00018},
  url={https://doi.ieeecomputersociety.org/10.1109/SP63933.2026.00018}
}
```
