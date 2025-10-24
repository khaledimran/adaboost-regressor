# AdaBoost Regressor

Implementation of an **AdaBoost Regressor** model with automatic preprocessing (numeric imputation + one-hot encoding), 
cross-validation, evaluation, and model export for reuse.

---

## Overview
This repository is part of a shared machine learning model collection.  
It demonstrates a full, reproducible workflow for training, evaluating, and saving an AdaBoost Regressor pipeline.

---

## Features
- End-to-end pipeline with preprocessing, model training, and evaluation  
- Automatic handling of missing and categorical data  
- Cross-validation and hyperparameter tuning  
- Residual and diagnostic visualizations  
- Feature importance and partial dependence plots  
- Model and schema export (`.joblib` + `.json`) for inference reuse  

---

## Notebook Structure
1. **Imports and Configuration** – setup, helper functions, and reproducibility settings  
2. **Data Loading** – California Housing dataset with synthetic fallback  
3. **Train/Test Split** – clean separation to avoid data leakage  
4. **Baseline Model** – DummyRegressor benchmark  
5. **Build the Full AdaBoost Regression Pipeline (with Preprocessing)**  
6. **Fast Grid Search (CV on Subsample)** – hyperparameter tuning  
7. **Refit on Full Training Data**  
8. **Evaluation and Diagnostics** – metrics, residuals, learning curves  
9. **Explainability** – permutation importance and partial dependence plots  
10. **Model Saving and Schema Export** – full pipeline + schema for reuse  
11. **Inference Demo** – loading the pipeline and generating predictions  

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/adaboost-regressor.git
   cd adaboost-regressor
   ```
2. Open the notebook in **Jupyter** or **Google Colab**:
   ```bash
   adaboost_regressor.ipynb
   ```
3. Run all cells sequentially.  
   The trained model and schema will be saved to the `artifacts/` directory.  

---

## Outputs
After running, you’ll find:
- `artifacts/adaboost_pipeline.joblib` → full pipeline (preprocessing + model)  
- `artifacts/schema.json` → column order, target, data source info  
- Plots and feature importances saved as `.png` in the same folder  

---

## Requirements
- Python 3.10+  
- scikit-learn ≥ 1.2  
- pandas, numpy, matplotlib, scipy, joblib  

(You can install these using `pip install -r requirements.txt` if you add one.)

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Author
**Khaled Imran Khan**  
Course: Machine Learning Model Repository Project  
Year: 2025
