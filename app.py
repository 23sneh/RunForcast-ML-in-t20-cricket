# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 02:30:35 2021

@author: Sneh
"""


# Importing the dependencies
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost
from xgboost import XGBRegressor

# Lets load the model
pipe = pickle.load(open("pipe.pkl", "rb"))

# Top 10 teams
teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Afghanistan',
 'Pakistan',
 'Sri Lanka']

# Cities having bowled more than 600 balls
cities = ['Colombo',
 'Mirpur',
 'Johannesburg',
 'Dubai',
 'Auckland',
 'Cape Town',
 'London',
 'Pallekele',
 'Barbados',
 'Sydney',
 'Melbourne',
 'Durban',
 'St Lucia',
 'Wellington',
 'Lauderhill',
 'Hamilton',
 'Centurion',
 'Manchester',
 'Abu Dhabi',
 'Mumbai',
 'Nottingham',
 'Southampton',
 'Mount Maunganui',
 'Chittagong',
 'Kolkata',
 'Lahore',
 'Delhi',
 'Nagpur',
 'Chandigarh',
 'Adelaide',
 'Bangalore',
 'St Kitts',
 'Cardiff',
 'Christchurch',
 'Trinidad']

# Lets set the app title
st.title("Men's T20 Cricket Runs Predictor")

# Lets set the batting and bowling teams
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select batting team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select bowling team", sorted(teams))
    
# Lets set the city
city = st.selectbox("Select city", sorted(cities))

# Lets set the current_score, overs and wickets
col3, col4, col5  = st.columns(3)

with col3:
    current_score = st.number_input("Current score")
with col4:
    overs = st.number_input("Overs done(works for overs > 5)")
with col5:
    wickets = st.number_input("Wickets out")
  
# Lets set the last five overs runs    
last_five = st.number_input("Runs scored in last 5 overs")


if st.button('Predict Runs'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Runs - " + str(int(result[0])))