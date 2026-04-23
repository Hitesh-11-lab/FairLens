
# 🔍 FairLens – Bias Detection & Mitigation for AI Systems

FairLens helps organizations detect and fix hidden discrimination in datasets and machine learning models before they impact real people.

## Problem
AI models making decisions about jobs, loans, and healthcare often learn historical biases. FairLens provides an easy way to measure, flag, and fix unfairness.

## Features
- **Data Bias Detection** – Disparate Impact score with red/yellow/green alerts
- **Data Mitigation** – Reweighting & Threshold Adjustment (one-click fix)
- **Model Fairness Evaluation** – Upload a trained model, get Demographic Parity & Equalized Odds
- **Post‑Processing Fix** – Flip predictions to reduce bias, download corrected outputs
- **No‑Code Dashboard** – Built with Streamlit, accessible to non‑technical users

## How It Works
1. Upload a CSV dataset (e.g., loan applications)
2. Select sensitive attribute (gender, race) and outcome column
3. Run bias analysis – see if any group is disadvantaged
4. Click "Fix" – get a debiased dataset or fairer model predictions

## Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### Clone & Setup
```bash
git clone https://github.com/yourusername/fairlens.git
cd fairlens
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
