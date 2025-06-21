"""This module trains the model by applying GridSearchCV to find the best estimator."""

import joblib
import logging
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_cleaning import clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Cleaning data...")
data = clean_data()

# Features and Label
X = data.drop('FS_score', axis=1)
y = data['FS_score']

# Resample the data using SMOTE
logging.info("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing
logging.info("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled,
    stratify=y_resampled,
    test_size=0.2,
    random_state=42
)

# Standard scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial XGBoost model
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42,
)

# GridSearchCV - to find best estimator
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Start GridSearchCV
start_time = time.time()
logging.info("Starting GridSearchCV...")
grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_train_scaled, y_train)
logging.info(f"GridSearchCV completed in {(time.time() - start_time)/60:.2f} minutes")

# Save the scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(grid.best_estimator_, 'model.pkl')
logging.info("Model training complete.")
