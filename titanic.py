# D:
# path : Python Projects\Titanic_Streamlit
# env : titanic
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# https://emojipedia.org for emoji links

def load_text(path):
  '''
  Wrapper function to load flavour texts
  '''
  with open(path) as f:
    texts = f.readlines()
  return texts

st.set_page_config(
    page_title="Survive Titanic!",
    page_icon=":ocean:",
    layout="centered",
    initial_sidebar_state="auto")

# Title
st.title("Would you Survive Titanic?")
# Introduction
intro = load_text('Flavour Texts/introduction.txt')

st.header("")
st.write(intro[0])
col1, col2 = st.beta_columns(2)

image = Image.open('Image/jack and rose.jfif')
st.image(image, caption='Jack and Rose')

# Configuring Sidebar
st.sidebar.header('Please enter your information!')
Pclass_dict = {'1st Class': 1,'2nd Class': 2,'3rd Class': 3} # convert strings to class numbers

def user_input_features():
  sex = st.sidebar.selectbox('Gender',('male','female'))
  Pclass = st.sidebar.selectbox('Ticket Class',('1st Class','2nd Class','3rd Class'))
  Age = st.sidebar.number_input('Age(years)', 0,100,28)
  alone = st.sidebar.selectbox('Travelling Alone?',('Yes','No'))
  data = {'Pclass': Pclass_dict[Pclass],
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
def predict_result(data_df):
  proc_feat = preprocessor.transform(data_df)
  res = tree.predict_proba(proc_feat)
  res = res[:,1]

  return res

result = predict_result(input_df)
# Prediction
st.sidebar.header('Prediction')
st.sidebar.write(f"*You have a **{result[0]*100:.2f}%** chance of surviving Titanic!*")
# note < add flavour texts in the sidebar, looks better.
# st.beta_expander <- use this for extra details
# st.sidebar.write(f"You have a {result[0]*100:.2f}% chance of surviving Titanic!")

# Jack and Rose Details
jack = {'Pclass': 1,
          'Sex': 'male',                
          'Age': 20,
          'Is_alone': 0}

rose = {'Pclass': 1,
          'Sex': 'female',                
          'Age': 18,
          'Is_alone': 1}

jack_df = pd.DataFrame(jack, index=[0])
rose_df = pd.DataFrame(rose, index=[0])

jack_pred = predict_result(jack_df)
rose_pred = predict_result(rose_df)
# Load Jack and Rose Flavour text
j_and_r = load_text('Flavour Texts/jack and rose.txt')
# Jack's prediction
with col1:
  with st.beta_expander("See Jack's result!"):
    st.write(jack_pred[0])

# Rose's prediction
with col2:
  with st.beta_expander("See Rose's result!"):
    st.write(rose_pred[0])

# History Flavor Text
st.header('**History of Titanic**')
st.write("")

description = load_text('Flavour Texts/description.txt')
st.write(description[0])
st.write(description[1])
