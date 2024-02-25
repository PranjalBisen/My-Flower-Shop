import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import base64

# Function to train and save the model
def train_and_save_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    joblib.dump(model, 'svc_iris_model.pkl')

# Check if the model is already trained and saved
if not os.path.exists('svc_iris_model.pkl'):
    train_and_save_model()

# Load trained model
model = joblib.load('svc_iris_model.pkl')

# Load Iris dataset for labels and feature names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names
import streamlit as st
def set_bg_img(img_path):
    """
    A function to set a background image.
    Args:
    - img_path: The path to the background image file
    """
    with open(img_path, "rb") as file:
        btn_img = file.read()
    btn_img_base64 = base64.b64encode(btn_img).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{btn_img_base64}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_bg_img("img.jpg")
# Streamlit Setup
st.title("Welcome to Pranjal's Flower Shop (SVC Branch)")
st.write("Oops!, Did you forget the flower you wanted to buy")
st.write("No issues, Fill the form below and we'll help")

# User inputs
inputs = []
for feature in feature_names:
    # st.sidebar.slider('ABC',0,20,10)
    value = st.slider(f'Input for {feature}', float(np.min(iris.data[:, feature_names.index(feature)])), float(np.max(iris.data[:, feature_names.index(feature)])), float(np.mean(iris.data[:, feature_names.index(feature)])))
    inputs.append(value)

# Prediction
if st.button('Prediction'):
    pred = model.predict([inputs])[0]
    st.write(f'Predicted species: {target_names[pred]}')
    if (target_names[pred] == 'versicolor'):
        st.image('versicolor.jpg')
        #st.write('Thank You for Visiting')
    elif (target_names[pred] == 'setosa'):
        st.image('setosa.jpg')
    elif (target_names[pred] == 'virginica'):
        st.image('virginica.jpg')
    # Visualization
    st.subheader('Feature Importances (Simplified Visualization)')
    fig, ax = plt.subplots()
    ax.barh(feature_names, model.coef_[0])
    st.pyplot(fig)

#Img BG
page_bg_img = '''
<style>
body {
background-image: url("https://imgs.search.brave.com/49waejPMcGpTUrY6PoFMM72TXi-4a9KEOUftOGYngAw/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2YwL2E3/L2NmL2YwYTdjZjhj/NzcyZWY3MTY5ZjJh/YTY2ZjI0OTc2ZTBl/LmpwZw");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)