import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

stroke =  pd.read_csv('healthcare-dataset-stroke-data.csv').drop(labels=['id'], axis=1)

st.write("""
# Stroke Prediction
Based on input parameters such as gender, age, various diseases, and smoking status, this model is used to determine whether a patient is likely to have a stroke.
""")

def user_input_var():
    gender = st.radio("Enter your Gender",["Male", "Female", "Other"])
    age = st.slider("Age", 15,100, 40)
    hypertension = st.radio("Do you suffer from high blood pressure?", ["Yes","No"], 1)
    if hypertension == "Yes": hypertension = 1
    else: hypertension = 0
    heart_disease = st.radio("Do you suffer from heart disease?", ["Yes","No"], 1)
    if heart_disease == "Yes": heart_disease = 1
    else: heart_disease = 0
    ever_married = st.radio("Are you Married?", ["Yes", "No"], 1)
    work_type = st.radio("Enter your Work Type", ["Children", "Never Worked", "Self Employed", "Government Job", "Private Job"],2)
    if work_type == "Student": work_type = "children"
    elif work_type == "Never Worked": work_type = "Never_worked"
    elif work_type == "Self Employed": work_type = "Self-employed"
    elif work_type == "Government Job": work_type = "Govt_job"
    else: work_type = "Private"
    Residence_type = st.radio("Enter your Residence type", ["Urban", "Rural"], 0), 
    avg_glucose_level = st.slider("Enter your Average Glucose level", 50, 400, 80)
    bmi =  st.slider("Enter your BMI", 15, 50, 20)
    smoking_status = st.radio("Are you a Smoker?", ["Former Smoker", "Never Smoked", "Smokes"],1)
    if smoking_status == "Former Smoker": smoking_status = "formerly smoked"
    elif smoking_status == "Never Smoked": smoking_status = "never smoked"
    else: smoking_status = "smokes"
    data = {'gender': gender,
            'age': age, 
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married, 
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_var()

X = stroke.drop('stroke', axis='columns').values
y = stroke['stroke'].values

imputer = SimpleImputer(strategy='median')
imputer.fit(X[:,8:9])
X[:,8:9] = imputer.transform(X[:,8:9])

ct = make_column_transformer(
    (OneHotEncoder(), [0,4,5,6,9]),
    remainder='passthrough')
X = ct.fit_transform(X)

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, stratify=y)

model = RandomForestClassifier(n_estimators= 150, criterion= 'entropy', random_state=42)
model.fit(X_train,y_train)


if st.button('Predict'):
    df = ct.transform(df)
    prediction = model.predict(df)
    out = {0:"You are Healthy", 1:"Sorry, but you should see a physician because there's a risk you'll have a stroke."}
    st.write(out[int(prediction)])
else: pass
