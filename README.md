# AI/ML Engineering Internship - DevelopersHub Corporation

This project is a part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**, Islamabad.

## Internship Details

- **Company:** DevelopersHub Corporation, Islamabad 🇵🇰
- **Internship Period:** July - September 2025

# Customer Churn Prediction

## Objective
This project implements an end-to-end machine learning pipeline for predicting customer churn. The goal is to identify customers who are likely to discontinue using a service, enabling proactive retention strategies.

- View the app screenshots [here](./app-screenshots)
- Explore the app: 

## Features
- Data preprocessing and feature engineering
- Model training with hyperparameter tuning
- Web-based interface for predictions
- Input validation and error handling
- Model performance visualization

## Installation

1. Clone the repository:
   ```bash
   git clone 'repository s URL'
   cd customer-churn-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv churn-env
   source churn-env/bin/activate  # On Windows: churn-env\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Running the Web App**:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`

## Methodology / Approach

### Data

- Dataset source - [Kaggle](https://www.kaggle.com/datasets/smayanj/customer-churn-prediction-dataset)

- **Dataset**: The model is trained on a dataset containing 50,000 customer records with the following features:
  - `tenure_months`: Number of months as a customer
  - `monthly_usage_hours`: Average monthly usage hours
  - `has_multiple_devices`: Whether the customer uses multiple devices (0/1)
  - `customer_support_calls`: Number of support calls made
  - `payment_failures`: Number of payment failures
  - `is_premium_plan`: Whether the customer is on a premium plan (0/1)
  - `churn`: Target variable (0 = No churn, 1 = Churn)

### Preprocessing
- Handling missing values
- Feature scaling
- Encoding categorical variables
- Train-test split (80-20)

### Model Training
- Implemented using scikit-learn's Pipeline API
- Models evaluated:
  - Logistic Regression
  - Random Forest Classifier
- Hyperparameter tuning using GridSearchCV
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC

### Web Interface
- Built with Streamlit
- Input validation
- Real-time predictions
- Model performance visualization

## Key Results

### Model Performance

| **Metric**   | **Score** |
|--------------|-----------|
| Accuracy     | 0.984     |
| Precision    | 0.683     |
| Recall       | 0.406     |
| F1-Score     | 0.509     |
| AUC-ROC      | 0.974     |

### Key Observations
1. The Random Forest model outperformed Logistic Regression in terms of overall accuracy and AUC-ROC score.
2. The most important features for predicting churn were found to be:
   - `tenure_months`
   - `monthly_usage_hours`
   - `payment_failures`
3. The model shows good generalization with consistent performance on the test set.

## Project Structure
```
ChurnPredictor/
├── app-screenshots/             # Snapshots of the App
├── dataset/                     # Dataset directory
│   └── customer_churn_dataset.csv
├── .gitignore                  # Git ignore file
├── README.md                   # This file
├── app.py                     # Streamlit web application
├── customer_churn_pipeline.ipynb # Jupyter notebook for model development
├── best_churn_model_pipeline.pkl # Trained model pipeline
├── model_info.pkl              # Model metadata and validation rules
└── requirements.txt            # Dependencies
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn
- joblib