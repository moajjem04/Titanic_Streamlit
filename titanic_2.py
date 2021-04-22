# D:
# Python Projects\Titanic_Streamlit
# env : titanic
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title("Would you Survive Titanic?")

# Configuring Sidebar
st.sidebar.header('Please enter your information!')

def user_input_features():
    sex = st.sidebar.selectbox('Gender',('male','female'))
    Pclass = st.sidebar.selectbox('Ticket Class',(1,2,3))
    Age = st.sidebar.number_input('Age(years)', 0,100,28)
    alone = st.sidebar.selectbox('Travelling Alone?',('Yes','No'))
    data = {'Pclass': Pclass,
            'Sex': sex,                
            'Age': Age,
            'Is_alone': alone}
    features = pd.DataFrame(data, index=[0])
    return features

# Getting Input
input_df = user_input_features()
# Processing Input
input_df['temp'] = 1
input_df.loc[input_df['Is_alone'] == 'Yes', 'temp'] = 0
input_df['Is_alone'] = input_df['temp']
input_df.drop('temp',axis = 1,inplace= True)

# Reads in saved classification model
load_clf = pickle.load(open('tree.pkl', 'rb'))
preprocessor = load_clf['preprocess'] # scaling and one-hot encoding
tree = load_clf['tree_model']         # Classification model

# Prediction
proc_feat = preprocessor.transform(input_df)
result = tree.predict_proba(proc_feat)
result = result[:,1]

# Prediction
st.subheader('Prediction')
st.write(f"You have a {result[0]*100:.2f}% chance of surviving Titanic!")

# History Flavor Text
st.subheader('History of Titanic')
with open('description.txt') as f:
  lines = f.readlines()
st.write(lines[0])
st.write(lines[1])
