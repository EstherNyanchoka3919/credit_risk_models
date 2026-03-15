import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load model and artifacts
@st.cache_resource
def load_model():
    pipeline = joblib.load('credit_risk_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    num_cols = joblib.load('num_cols.pkl')
    return pipeline, preprocessor, num_cols

pipeline, preprocessor, num_cols = load_model()
classifier = pipeline.named_steps['classifier']

st.set_page_config(page_title='Credit Risk Assessor', layout='wide')
st.title('🏦 Credit Risk Assessment Tool')
st.markdown('Enter applicant details below to get an instant risk evaluation.')

# Create input form (two columns)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['M', 'F'])
    own_car = st.selectbox('Owns a car?', ['Y', 'N'])
    own_property = st.selectbox('Owns property?', ['Y', 'N'])
    children = st.number_input('Number of children', 0, 19, 0)
    income = st.number_input('Annual income', 20000, 1000000, 50000)
    income_type = st.selectbox('Income type', ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
    education = st.selectbox('Education', ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'])

with col2:
    family_status = st.selectbox('Family status', ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
    housing = st.selectbox('Housing type', ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'])
    phone = st.selectbox('Has phone?', [0, 1])
    email = st.selectbox('Has email?', [0, 1])
    occupation = st.selectbox('Occupation', ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Secretaries', 'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff', 'Unknown'])
    family_size = st.number_input('Family size', 1, 20, 2)
    age = st.number_input('Age', 18, 100, 35)
    emp_years = st.number_input('Years employed', 0.0, 50.0, 5.0)

# Create a DataFrame for the input
input_dict = {
    'GENDER': gender,
    'OWN_CAR': own_car,
    'OWN_REALTY': own_property,
    'CHILDREN': children,
    'ANNUAL_INCOME': income,
    'INCOME_TYPE': income_type,
    'EDUCATION': education,
    'FAMILY_STATUS': family_status,
    'HOUSING': housing,
    'PHONE': phone,
    'EMAIL': email,
    'OCCUPATION': occupation,
    'FAMILY_SIZE': family_size,
    'AGE': age,
    'EMPLOYED_YEARS': emp_years
}

# Derive engineered features
input_dict['INCOME_PER_CAPITA'] = income / max(family_size, 1)
input_dict['EMPLOYMENT_STABILITY'] = emp_years / max(age - 18, 1)
input_dict['AGE_INCOME'] = age * income / 1e6

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One‑hot encode (must match training columns)
# We need to replicate the one‑hot transformation – easier to use the preprocessor pipeline.
# The preprocessor only scales numeric columns; one‑hot was done manually.
# So we must apply the same dummy encoding as in training.

# For simplicity, we'll load a full preprocessor that includes encoding.
# But here we will reconstruct using the training columns list.
# (In practice, you would save the full ColumnTransformer with encoder.)

# Manually apply dummies (same as notebook)
input_encoded = pd.get_dummies(input_df, columns=['GENDER','INCOME_TYPE','EDUCATION','FAMILY_STATUS','HOUSING','OCCUPATION'], drop_first=True)
# Ensure all columns present in training are present (fill missing with 0)
missing_cols = set(num_cols) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[num_cols]  # order matches

# Scale
input_scaled = preprocessor.transform(input_encoded)

# Predict
prob = pipeline.named_steps['classifier'].predict_proba(input_scaled)[0,1]
pred = (prob >= 0.3).astype(int)

# Display results
st.markdown('---')
st.header('Risk Assessment Result')

col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    st.metric('Default Probability', f'{prob:.2%}')
with col_res2:
    decision = '❌ REJECT' if pred == 1 else '✅ APPROVE'
    st.metric('Decision', decision)
with col_res3:
    st.metric('Risk Category', 'High' if prob > 0.5 else 'Medium' if prob > 0.2 else 'Low')

# SHAP Explanation
if st.button('Explain this decision'):
    st.subheader('SHAP Force Plot – Why this decision?')
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(input_scaled)
    # Plotly doesn't have native waterfall, so use matplotlib
    fig, ax = plt.subplots(figsize=(10,4))
    shap.force_plot(explainer.expected_value, shap_values[0], input_scaled[0], feature_names=num_cols, matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    # Feature importance bar chart
    st.subheader('Top Factors')
    shap_df = pd.DataFrame({'feature': num_cols, 'shap_value': shap_values[0]})
    shap_df['abs'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs', ascending=False).head(10)
    fig2 = px.bar(shap_df, x='shap_value', y='feature', orientation='h',
                  title='SHAP values (impact on model output)',
                  labels={'shap_value':'SHAP value', 'feature':''})
    st.plotly_chart(fig2)

# Risk meter (optional)
st.markdown('---')
st.subheader('Risk Profile')
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob * 100,
    domain = {'x': [0,1], 'y': [0,1]},
    title = {'text': "Default Risk (%)"},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "darkred" if prob>0.5 else "orange" if prob>0.2 else "green"},
             'steps': [
                 {'range': [0, 20], 'color': "lightgreen"},
                 {'range': [20, 50], 'color': "yellow"},
                 {'range': [50, 100], 'color': "salmon"}],
             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prob*100}}))
st.plotly_chart(fig_gauge)
