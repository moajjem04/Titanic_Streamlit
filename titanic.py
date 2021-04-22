# D:\Python Projects\Titanic_Streamlit
# env : titanic
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Titanic App
""")

st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

# Collects user input features into dataframe
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
def user_input_features():
    sex = st.sidebar.selectbox('Gender',('male','female'))
    Pclass = st.sidebar.selectbox('Ticket Class',(1,2,3))
    Age = st.sidebar.slider('Age(years)', 0,100,28,1)
    alone = st.sidebar.selectbox('Travelling Alone?',('Yes','No'))
    data = {'Pclass': Pclass,
            'Sex': sex,                
            'Age': Age,
            'Is_alone': alone}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

input_df['temp'] = 1
input_df.loc[input_df['Is_alone'] == 'Yes', 'temp'] = 0
input_df['Is_alone'] = input_df['temp']
input_df.drop('temp',axis = 1,inplace= True)


# Reads in saved classification model
load_clf = pickle.load(open('tree.pkl', 'rb'))

preprocessor = load_clf['preprocess']
tree = load_clf['tree_model']

proc_feat = preprocessor.transform(input_df)
result = tree.predict_proba(proc_feat)
result = result[:,1]

st.subheader('Prediction')
st.write(f"You have a {result[0]*100:.2f}% chance of surviving Titanic!")

# # Apply model to make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)


# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)