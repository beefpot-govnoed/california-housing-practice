# California Housing Price Prediction (Ensemble ML)

California housing median price prediction project. Covers the full ML development pipeline: from baseline model to advanced Feature Engineering and gradient boosting ensembling.

---

## Results

During experiments, model accuracy (R2) was improved from 80.50% to 88.14%, while Mean Absolute Error (MAE) dropped from $32,754 to ~$23,200.

| Stage / Model | MAE ($) | R2 Score (%) |
| :--- | :---: | :---: |
| Baseline (RandomForest) | $32,754 | 80.50% |
| Baseline + Dist_to_SF | $31,840 | 81.20% |
| XGBoost (Single) | $28,306 | 85.20% |
| Ensemble (XGB + Cat + Geo-Features) | $23,200 | 88.14% |

Note: The ~88% threshold is close to the theoretical maximum for this dataset due to block-group data aggregation (1990 Census) and missing information on property repair conditions.

---

## Key Improvement Methods (Feature Engineering)

1. Proximity to Ocean (Dist_to_Coast): Approximate distance calculation to the Pacific Ocean using coordinates (primary price factor).
2. Geographical Distances: Distances to major metropolitan areas (San Francisco, Los Angeles, San Jose, San Diego).
3. Spatial Clustering: K-Means usage (35 clusters) and grid rounding for precise location mapping.
4. Capping (Outlier Handling): Enforcing strict prediction bounds [0.15, 5.0] according to the original dataset range.
5. Ensembling (Blending): Weighted combination of XGBoost and CatBoost predictions with automatic selection of the best-performing mix.

---

## Getting Started

### 1. Clone the repository
git clone https://github.com/beefpot-govnoed/california-housing-practice.git
cd california-housing-practice

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the script
python california-housing-practice.py

---

## Git Commands for Uploading

git init
git add .
git commit -m "Initial commit: California Housing 88.14% R2 ensemble"
git branch -M main
git remote add origin https://github.com/beefpot-govnoed/california-housing-practice.git
git push -u origin main
