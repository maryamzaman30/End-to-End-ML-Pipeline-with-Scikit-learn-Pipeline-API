import streamlit as st
import pandas as pd
import joblib

# Load the exported model pipeline and model info
@st.cache_resource
def load_model_and_info():
    model = joblib.load('best_churn_model_pipeline.pkl')
    model_info = joblib.load('model_info.pkl')
    return model, model_info

model, model_info = load_model_and_info()

# App title and description
st.title('Customer Churn Prediction App')
st.markdown('''
This app uses a machine learning model to predict if a customer is likely to churn.
Enter the customer details below and click "Predict".
''')

# Display model information
st.subheader('Model Information')
st.write(f"**Model Name**: {model_info['model_name']}")
st.write(f"**Model Type**: {model_info['model_type']}")
st.write(f"**Test Performance**:")
st.write(f"- Accuracy: {model_info['metrics']['accuracy']:.3f}")
st.write(f"- Precision: {model_info['metrics']['precision']:.3f}")
st.write(f"- Recall: {model_info['metrics']['recall']:.3f}")
st.write(f"- F1-Score: {model_info['metrics']['f1']:.3f}")
st.write(f"- AUC-ROC: {model_info['metrics']['auc']:.3f}")

# Fallback validation rules if not present in model_info
default_validation_rules = {
    'feature_rules': {
        'tenure_months': {
            'dtype': 'int64',
            'min': 0,
            'max': 100,
            'description': 'Number of months the customer has been with the service'
        },
        'monthly_usage_hours': {
            'dtype': 'float64',
            'min': 0.0,
            'max': 100.0,
            'description': 'Average monthly usage hours'
        },
        'has_multiple_devices': {
            'dtype': 'int64',
            'allowed_values': [0, 1],
            'description': 'Whether the customer uses multiple devices (0=No, 1=Yes)'
        },
        'customer_support_calls': {
            'dtype': 'int64',
            'min': 0,
            'max': 10,
            'description': 'Number of customer support calls made'
        },
        'payment_failures': {
            'dtype': 'int64',
            'min': 0,
            'max': 5,
            'description': 'Number of payment failures'
        },
        'is_premium_plan': {
            'dtype': 'int64',
            'allowed_values': [0, 1],
            'description': 'Whether the customer is on a premium plan (0=No, 1=Yes)'
        }
    }
}

# Use validation rules from model_info if available, else use default
validation_rules = model_info.get('validation_rules', default_validation_rules)

# Function to validate inputs based on validation_rules
def validate_input(feature, value, rules):
    rule = rules['feature_rules'][feature]
    if rule['dtype'] == 'int64':
        if not isinstance(value, (int, float)) or value != int(value):
            return f"{feature}: Must be an integer"
        if 'min' in rule and value < rule['min']:
            return f"{feature}: Must be >= {rule['min']}"
        if 'max' in rule and value > rule['max']:
            return f"{feature}: Must be <= {rule['max']}"
        if 'allowed_values' in rule and value not in rule['allowed_values']:
            return f"{feature}: Must be one of {rule['allowed_values']}"
    elif rule['dtype'] == 'float64':
        if not isinstance(value, (int, float)):
            return f"{feature}: Must be a number"
        if 'min' in rule and value < rule['min']:
            return f"{feature}: Must be >= {rule['min']}"
        if 'max' in rule and value > rule['max']:
            return f"{feature}: Must be <= {rule['max']}"
    return None

# Input form for features with validation
with st.form(key='prediction_form'):
    st.subheader('Enter Customer Details')
    tenure_months = st.number_input(
        f"Tenure (months) - {validation_rules['feature_rules']['tenure_months']['description']}",
        min_value=float(validation_rules['feature_rules']['tenure_months']['min']),
        max_value=float(validation_rules['feature_rules']['tenure_months']['max']),
        value=30.0,
        step=1.0
    )
    monthly_usage_hours = st.number_input(
        f"Monthly Usage Hours - {validation_rules['feature_rules']['monthly_usage_hours']['description']}",
        min_value=validation_rules['feature_rules']['monthly_usage_hours']['min'],
        max_value=validation_rules['feature_rules']['monthly_usage_hours']['max'],
        value=20.0,
        step=0.1
    )
    has_multiple_devices = st.selectbox(
        f"Has Multiple Devices? - {validation_rules['feature_rules']['has_multiple_devices']['description']}",
        validation_rules['feature_rules']['has_multiple_devices']['allowed_values']
    )
    customer_support_calls = st.number_input(
        f"Customer Support Calls - {validation_rules['feature_rules']['customer_support_calls']['description']}",
        min_value=float(validation_rules['feature_rules']['customer_support_calls']['min']),
        max_value=float(validation_rules['feature_rules']['customer_support_calls']['max']),
        value=1.0,
        step=1.0
    )
    payment_failures = st.number_input(
        f"Payment Failures - {validation_rules['feature_rules']['payment_failures']['description']}",
        min_value=float(validation_rules['feature_rules']['payment_failures']['min']),
        max_value=float(validation_rules['feature_rules']['payment_failures']['max']),
        value=0.0,
        step=1.0
    )
    is_premium_plan = st.selectbox(
        f"Is Premium Plan? - {validation_rules['feature_rules']['is_premium_plan']['description']}",
        validation_rules['feature_rules']['is_premium_plan']['allowed_values']
    )
    
    submit_button = st.form_submit_button(label='Predict')

# Validate inputs and make prediction when form is submitted
if submit_button:
    # Create input dictionary
    input_data = {
        'tenure_months': tenure_months,
        'monthly_usage_hours': monthly_usage_hours,
        'has_multiple_devices': has_multiple_devices,
        'customer_support_calls': customer_support_calls,
        'payment_failures': payment_failures,
        'is_premium_plan': is_premium_plan
    }
    
    # Validate inputs
    validation_errors = []
    for feature, value in input_data.items():
        error = validate_input(feature, value, validation_rules)
        if error:
            validation_errors.append(error)
    
    if validation_errors:
        st.error("Input validation errors:")
        for error in validation_errors:
            st.error(error)
    else:
        # Create DataFrame from inputs (must match feature order from training)
        input_df = pd.DataFrame([input_data], columns=model_info['feature_names'])
        
        # Predict and get probability
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display results
            st.subheader('Prediction Result')
            if prediction == 1:
                st.error(f'The customer is likely to **CHURN** (Probability: {probability:.2%})')
            else:
                st.success(f'The customer is likely to **STAY** (Probability of churn: {probability:.2%})')
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")