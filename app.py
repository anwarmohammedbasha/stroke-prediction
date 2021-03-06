import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

stroke =  pd.read_csv('healthcare-dataset-stroke-data.csv').drop(labels=['id'], axis=1)
stroke.drop(stroke[stroke.gender=='Other'].index, inplace=True)
stroke.replace(np.nan, stroke.bmi.median(),inplace=True)
stroke.smoking_status.replace({'never smoked':0,'Unknown':np.nan,'formerly smoked':1,'smokes':1}, inplace=True)
stroke.smoking_status.replace(np.nan, stroke.smoking_status.value_counts().argmax(), inplace=True)
stroke.work_type.replace({'Never_worked':'Student',"children":'Student',
                         "Private":'Private-Job', 'Self-employed':'Self-Employed',
                         'Govt_job':'Govt-Job'},inplace=True)

st.write("""
# Stroke Prediction
Based on input parameters such as gender, age, various diseases, and smoking status, this model is used to determine whether a patient is likely to have a stroke.
""")

def user_input_var():
    gender = st.radio("Enter your Gender",["Male", "Female"])
    age = st.slider("Age", 18, 70, 25)
    hypertension = st.radio("Do you suffer from high blood pressure?", ["Yes","No"], 1)
    if hypertension == "Yes": hypertension = 1
    else: hypertension = 0
    heart_disease = st.radio("Do you suffer from heart disease?", ["Yes","No"], 1)
    if heart_disease == "Yes": heart_disease = 1
    else: heart_disease = 0
    ever_married = st.radio("Are you Married?", ["Yes", "No"], 1)
    work_type = st.radio("Enter your Work Type", ["Student", "Self Employed", "Government Job", "Private Job"],2)
    if work_type == "Student": work_type = "Student"
    elif work_type == "Self Employed": work_type = "Self-Employed"
    elif work_type == "Government Job": work_type = "Govt-Job"
    else: work_type = "Private-Job"
    residence_type = st.radio("Enter your Residence type", ["Urban", "Rural"], 0), 
    avg_glucose_level = st.slider("Enter your Average Glucose level", 50, 400, 80)
    bmi =  st.slider("Enter your BMI", 15, 50, 20)
    smoking_status = st.radio("Are you a Smoker?", ["Yes", "No"],1)
    if smoking_status == "Yes": smoking_status = 1
    else: smoking_status = 0
    data = {'gender': gender,
            'age': age, 
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married, 
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_var()

X = stroke.drop('stroke', axis='columns').values
y = stroke['stroke'].values

ct = make_column_transformer(
    (OneHotEncoder(), [0,4,5,6]),
    remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 32, stratify=y)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train,y_train)

if st.button('Predict'):
    df = ct.transform(df)
    prediction = model.predict(df)
    out = {0:"You are Healthy", 1:"Sorry, but you should see a physician because there's a risk you'll have a stroke."}
    st.write(out[int(prediction)])
else: pass
