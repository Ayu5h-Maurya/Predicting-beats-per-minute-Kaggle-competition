# BPM Prediction - Kaggle Playground Series

A comprehensive machine learning pipeline for predicting song beats-per-minute using ensemble methods with LightGBM, CatBoost, and advanced feature engineering.

## ðŸŽ¯ Overview

This project implements a robust tabular ML pipeline for the Kaggle Playground Series S5E9 competition, focusing on predicting song BPM using ensemble methods and advanced feature engineering techniques.

## ðŸš€ Key Features

- **Multi-Model Ensemble**: LightGBM (CPU), CatBoost (GPU), XGBoost (CPU)
- **Advanced Feature Engineering**: Log transforms, interactions, statistical features
- **Cross-Validation**: Stratified K-Fold with configurable bins
- **Hyperparameter Tuning**: Optuna-based optimization for LightGBM and CatBoost
- **Stacking**: Ridge meta-model for improved predictions
- **Seed Ensembling**: Multiple seed averaging for stability
- **Clean Feature Engineering**: Minimal, high-signal feature sets
- **KMeans & PCA**: Cluster and dimensionality reduction features

## ðŸ› ï¸ Installation

1. **Clone the repository**
   `ash
   git clone <repository-url>
   cd bpm-prediction
   `

2. **Create virtual environment**
   `ash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   `

3. **Install dependencies**
   `ash
   pip install -r requirements.txt
   `

4. **Install PyTorch with CUDA support** (for GPU acceleration)
   `ash
   pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   `

## ðŸŽ® Usage

### Basic Training
`ash
python main.py --use-cat-gpu --out submission.csv
`

### Advanced Training with Feature Engineering
`ash
python main.py --use-cat-gpu --engineer-features --fe-clean --search-blend --out submission_advanced.csv
`

### Full Pipeline with Tuning and Ensembling
`ash
python main.py --use-cat-gpu --engineer-features --fe-clean --no-xgb --search-blend --cv-bins 20 --seed-ensemble 3 --kmeans 8,16 --pca 10 --tune-lgbm --tune-cat --tune-trials 80 --save-oof --save-importances --out submission_full.csv
`

## ðŸ† Best Results

Our best configuration achieves:
- **OOF RMSE**: ~26.458
- **Models**: LightGBM (60%) + CatBoost (40%)
- **Features**: Clean engineered features + KMeans + PCA
- **CV**: 10-fold stratified with 20 bins
- **Ensemble**: 3-seed averaging

## ðŸ“ˆ Model Performance

| Model | OOF RMSE | Training Time | Notes |
|-------|----------|----------------|-------|
| LightGBM (CPU) | 26.458 | ~6s | Best single model |
| CatBoost (GPU) | 26.459 | ~24s | Strong second |
| XGBoost (CPU) | 26.607 | ~140s | Often excluded from blend |
| Ensemble | 26.458 | ~30s | 60% LGBM + 40% CatBoost |

## ðŸŽ¯ Key Insights

1. **Feature Importance**: TrackDurationMs, RhythmScore, Energy, MoodScore are most important
2. **Model Selection**: LightGBM + CatBoost ensemble works best
3. **Feature Engineering**: Clean, minimal features outperform complex ones
4. **Cross-Validation**: 10-fold with 20 bins provides stable estimates
5. **Ensembling**: 3-seed averaging improves stability

## ðŸ“ Dependencies

See requirements.txt for full list. Key packages:
- pandas, numpy, scikit-learn
- lightgbm, catboost, xgboost
- optuna (hyperparameter tuning)
- tqdm (progress bars)

## ðŸ¤ Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ðŸ“„ License

This project is for educational and competition purposes.

## ðŸ™ Acknowledgments

- Kaggle Playground Series for the dataset
- LightGBM, CatBoost, XGBoost teams for excellent libraries
- Optuna for hyperparameter optimization
