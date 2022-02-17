# Would you Survive Titanic? :ocean:
![Heroku](https://pyheroku-badge.herokuapp.com/?app=survive-titanic&style=flat)

Here is the [streamlit app](https://survive-titanic.herokuapp.com) deployed in Heroku!

[//]: # (heroku link - https://survive-titanic.herokuapp.com)

|![Main Screen](Image/home%20screen.PNG?raw=true)|
|:--:|
|**The home screen of the webapp.**|
---

Bring out the time machine and travel back to 1912! 
Go on Titanic's maiden voyage and see if you would survive its *titanic* crash.

This [app](https://survive-titanic.herokuapp.com) uses a Decision Tree trained on the [Titanic Dataset](https://www.kaggle.com/c/titanic) to predict whether you will survive or not.

The machine learning model was trained in [this](https://www.kaggle.com/moajjem04/titanic-streamlit) Kaggle Notebook.

The aim of this project is to make a webapp using [Streamlit](https://survive-titanic.herokuapp.com) and deploy it in [Heroku](https://www.heroku.com).

This is why only **four features** are chosen for the machine learning model to learn:

1. `Pclass` (*Ticket Class*) : This variable denotes the class of the ticket for Titanic ship. The options are `1st Class`, `2nd Class` and `3rd Class`.
2. `Sex` : This indicates the sex of the person boarding the ship. The options are `male` and `female`.
3. `Age` : This variable is the age of the person in years. Only integer values are accepted and this ranges from `0` to `100`.
4. `Is_alone` : This variable denotes whether the person is travelling alone or not. The options are simply `Yes` and `No`.

The features were carefully chosen so that the user does not have any security concerns. 
To my best knowledge, no data is collected via this app.

Huge shout out to [Data Professor](https://www.youtube.com/c/DataProfessor) for his [video](https://youtu.be/zK4Ch6e1zq8) on how to deploy machine learning models to Heroku.

I would appreciate any feedback regarding this project.

Do check out the [app](https://survive-titanic.herokuapp.com) and have fun with it!

*p.s. Not liable if you drown in Titanic despite the model's prediction. :sweat_smile:*

---

# Some Screenshots from the app


|![Results for Jack and Rose](Image/jack_rose_result.PNG?raw=true)|
|:--:|
|**Predictions by the model on the chances of survival for Jack and Rose.**|

|![Model taking input](Image/ML.PNG?raw=true)|
|:--:|
|**Taking Inputs for the model to make predictions on.**|

