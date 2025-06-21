# Food Security Status Analysis
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.readthedocs.io/)

## Project Overview
This project carries out a machine learning analysis of household food security status. It includes a web application that predicts household food security scores based on socio-economic and demographic data. The model was developed using data collected from urban informal settlements in Nairobi, Kenya (2014).

## Table of Contents
- [Application Overview](#application-overview)
    - [How it Works](#how-it-works)
- [Installation and Setup](#installation-and-setup)
    - [Pre-requisites](#pre-requisites)
    - [Setup Instructions](#setup-instructions)
- [Basic Usage](#basic-usage)
- [Repository Structure](#repository-structure)
- [XGBoost Model Performance](#xgboost-model-performance)
- [Data Source](#data-source)
- [License](#license)

## Application Overview
This application allows users to input household characteristics (e.g. household size, education, wealth index) and receive a predicted Food Security Score (FS_score) ranging from 0 (secure) to 4 (severely insecure), along with class probabilities.

It was developed using:
- XGBoost Classifier
- SMOTE for handling class imbalance
- GridSearchCV for model optimization
- Streamlit for web interface

Further explanation on the reason why XGBoost was used, and analysis of the data can be found in `notebooks/food_security_status.ipynb`.

### How It Works
- Users provide demographic data via the UI
- The model preprocesses and scales input features
- XGBoost predicts a FS_score class and outputs probabilities for each class

## Installation and Setup
### Pre-requisites
- Python 3.12 or later

### Setup Instructions
- Clone the repository:
```bash
git clone https://github.com/Fidelisaboke/food-security-analysis.git
cd food-security-analysis
```

- Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

- Navigate to `pipeline/` and run `train_model.py` to train the XGBoost classifier:
```
cd pipeline
python train_model.py
```
**Warning:** This may take several minutes. Go grab some coffee and inspect the console output while it happens!

- Once model training is done, ensure `model.pkl` and `scaler.pkl` files are generated. This prevents the need for retraining the model if adjustments are not being made.
- You may return to the project's root directory once this process ends:
```
cd ..
```

## Basic Usage
Once installed, you can launch the Streamlit app:
```bash
cd app
streamlit run app.py
```
- The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

## Repository Structure
```
├── .gitignore
├── README.md
├── app
│   ├── app.py
│   └── utils.py
├── data
│   └── dataFS.csv
├── notebooks
│   └── food_security_status.ipynb
├── pipeline
│   ├── data_cleaning.py
│   └── train_model.py
└── requirements.txt
```

## XGBoost Model Performance
The table below shows the performance metrics of the XGBoost model used as the estimator:

| Metric       | Value |
| ------------ | ----- |
| Accuracy     | 70%   |
| F1-Score     | 0.70  |
| Macro Avg F1 | 0.70  |
| Weighted F1  | 0.70  |

**Note**: Results were achieved using XGBoost + SMOTE + tuning

## Data Source
This model was trained on anonymized household survey data collected in Nairobi’s informal settlements (2014), focusing on key social determinants of food security.

## License
This project is open-source software licensed under the [MIT License](https://opensource.org/license/MIT).
