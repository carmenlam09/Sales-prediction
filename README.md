# Sales Prediction Project

## Overview
This project analyzes a sales dataset and builds regression models to predict the total amount spent on each order using customer, product, and marketing data. The notebook includes data exploration, visualization, preprocessing, feature engineering, model training, and evaluation.

## Group Members
- Carmen Lam Kah Man
- Dernice Lee Tian Yi
- Lee Zia Qian
- Tan Teck Hou
- Wong Yi Fei

## Dataset
- Source: [Kaggle - Dmart Ready Online Store Dataset](https://www.kaggle.com/datasets/praneethkumar007/dmart-ready-online-store)

## Environment / Dependencies
The notebook installs and uses several Python packages, including:
- `pydantic==1.10.12`
- `ydata-profiling`
- `autoviz`
- `catboost`
- `tensorflow` / `keras`
- `keras-tuner`
- `h2o`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`

## Workflow

### 1. Data Loading and Exploration
- Load the dataset  
- Display summary statistics  
- Generate automated profile report with `ydata_profiling.ProfileReport`  
- Plot category counts and treemap of ratings by marketing channel  
- Create additional visualizations with `AutoViz`

### 2. Data Preprocessing
- Convert date columns (`Order Date`, `Delivery Date`, `Cancellation Date`)  
- Convert `Shipping Charges` to numeric and replace invalid values with `0`  
- Inspect missing values and analyze null rows  
- Detect duplicates and count unique values

### 3. Feature Engineering
- Create new features:  
  - `Discount Percentage` = `(MRP - Discount Price) / MRP`  
  - `Profit Margin` = `MRP - Discount Price`  
  - `Avg Time Per Click` = `Time Spent on Website / No of Clicks`  
  - `Is Repeat Customer` = duplicated `Customer ID`  
- Handle missing and infinite values with replacement and `fillna(0)`

### 4. Train / Validation / Test Split
- Split data into training, validation, and test sets using `train_test_split`  
- Define categorical and numerical columns  
- Build preprocessing pipeline with `ColumnTransformer`  
  - One-hot encode categorical features  
  - Standard scale numerical features  
- Transform train, validation, and test data

### 5. Model Training and Evaluation
The processed data is used to train and evaluate multiple regression models:

- **Decision Tree Regressor**  
  Baseline training and hyperparameter tuning with `GridSearchCV`.

- **Random Forest Regressor**  
  Baseline training and hyperparameter tuning with `GridSearchCV`.

- **Neural Network (Keras)**  
  - Sequential dense model compiled with Adam optimizer and MSE loss  
  - Trained for 20 epochs with validation split  
  - Tuned using `keras-tuner.RandomSearch`  
  - Retrained best model for 50 epochs

- **CatBoost Regressor**  
  Baseline training and hyperparameter tuning with `GridSearchCV`.

- **H2O Gradient Boosting Machine (GBM)**  
  - Initialized H2O environment  
  - Converted pandas DataFrames to H2O frames  
  - Trained GBM model with grid search  
  - Evaluated best model on test data

### 6. Results
All models are evaluated using standard regression metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

---

## Summary
This workflow provides a complete pipeline: from data exploration and preprocessing to feature engineering, model training, and evaluation. It compares traditional machine learning models, ensemble methods, and deep learning approaches, highlighting their performance across multiple metrics.
