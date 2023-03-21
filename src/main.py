#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

st.set_page_config(layout="wide")

# Functions for each of the pages
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')
    

def data_summary(df1,df2):
    st.header('Statistics of Dataframe')
    st.write(df1)
    st.write(df2)
    # st.write(df.describe())

def data_header():
    st.header('Header of Dataframe')
    # st.write(df1.head())

def predictive_modeling():
    st.header('Plot of Data')




def team_data(df1):
    #USE SCATTER PLOT
    col1, col2 = st.columns(2)
    
    sorted_unique_team = sorted(df1.TEAM.unique())
    team_option = ['All Teams'] + sorted_unique_team
    selected_teams = st.multiselect('Team', team_option, default='All Teams')

    try:
        if not selected_teams:
            raise ValueError('Please select at least one team.')
        elif 'All Teams' in selected_teams:
            df_selected_teams = df1.copy()
        else:
            df_selected_teams = df1[df1.TEAM.isin(selected_teams)].copy()
    except ValueError as e:
        st.warning(str(e))
        return
    
    x_axis_val = col1.selectbox('Select the X-axis', options=df_selected_teams.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df_selected_teams.columns)

    plot = px.scatter(df_selected_teams, x=x_axis_val, y=y_axis_val, color = df_selected_teams.TEAM, trendline='ols',
                      trendline_color_override='green', hover_name = "TEAM", hover_data=["YEAR", "W"] )
    
    # Add linear regression prediction for next season
    if x_axis_val == 'YEAR' and y_axis_val != 'YEAR':
        lr = LinearRegression()
        X = df_selected_teams[df_selected_teams.YEAR < 2022][[x_axis_val]]
        y = df_selected_teams[df_selected_teams.YEAR < 2022][[y_axis_val]]
        lr.fit(X, y)
        next_season = 2022
        next_year = np.array([next_season]).reshape(-1, 1)
        next_stat = lr.predict(next_year)
        st.write(f"Predicted {y_axis_val} for {next_season}: {next_stat[0][0]:.2f}")
        plot.add_trace(
            go.Scatter(
                x=[next_season],
                y=next_stat[0],
                mode='markers',
                name='Next season prediction',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle'
                )
            )
        )
        # Evaluation metrics
        y_pred = lr.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean squared error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")
    
    st.plotly_chart(plot, use_container_width=True)


def player_data(df2):
    col3, col4, col5 = st.columns(3)
    
    sorted_unique_team = sorted(df2.TEAM.unique())
    teams_option = ['All Teams'] + sorted_unique_team
    selected_teams = col5.multiselect('Team', teams_option, default='All Teams')

    if 'All Teams' in selected_teams:
        filtered_data_teams = df2.copy()
    else:
        filtered_data_teams = df2[df2.TEAM.isin(selected_teams)].copy()

    players = filtered_data_teams['Player'].unique()
    selected_player = col5.selectbox("Select player", players)
    filtered_data_players = filtered_data_teams[filtered_data_teams['Player'] == selected_player]

    x_axis_val = col3.selectbox('Select the X-axis', options=filtered_data_players.columns, key="3")
    y_axis_val = col4.selectbox('Select the Y-axis', options=filtered_data_players.columns, key="4")

    plot = px.scatter(filtered_data_players, x=x_axis_val, y=y_axis_val, color='TEAM', trendline='ols',
                      trendline_color_override='green', hover_name="Player", hover_data=["TEAM", "YEAR"])

    # Add linear regression prediction for next season
    if x_axis_val == 'YEAR' and y_axis_val in [y_axis_val]:
        lr = LinearRegression()
        X = filtered_data_players[filtered_data_players.YEAR < 2022][[x_axis_val]]
        y = filtered_data_players[filtered_data_players.YEAR < 2022][[y_axis_val]]
        lr.fit(X, y)
        next_season = 2022
        next_year = np.array([next_season]).reshape(-1, 1)
        next_stat = lr.predict(next_year)
        st.write(f"Predicted {y_axis_val} for {next_season}: {next_stat[0][0]:.2f}")
        plot.add_trace(
            go.Scatter(
                x=[next_season],
                y=next_stat[0],
                mode='markers',
                name='Next season prediction',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle'
                )
            )
        )
        # Evaluation metrics
        y_pred = lr.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean squared error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")
    
    st.plotly_chart(plot, use_container_width=True)

    # results = px.get_trendline_results(plot)
    # st.text(results)

    



# Add a title and intro text
st.title('NBA Data Explorer')
st.text('This is a web app to allow exploration of NBA Data')

# Sidebar setup
st.sidebar.title('Sidebar')
upload_file1 = st.sidebar.file_uploader('Upload a file containing NBA data', key="1")
upload_file2 = st.sidebar.file_uploader('Upload a file containing NBA data', key ="2")
#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Summary', 'Data Header', 'Scatter Plot', 'Interactive Plots'])

# Check if file has been uploaded
if upload_file1 is not None:
    df1 = pd.read_csv(upload_file1)
    # mapping = {player: i for i, player in enumerate(df1['TEAM'].unique())}
    # df1['TEAM'] = df1['TEAM'].map(mapping)
if upload_file2 is not None:
    df2 = pd.read_csv(upload_file2)
    # mapping = {player: i for i, player in enumerate(df2['Player'].unique())}
    # df2['Player'] = df2['Player'].map(mapping)
    # df = df.astype({'YEAR':'int'})
    # df.YEAR = pd.DatetimeIndex(df.YEAR).strftime("%Y")

# Navigation options
if options == 'Home':
    home(upload_file1)
elif options == 'Data Summary':
    data_summary()
elif options == 'Data Header':
    data_header()
elif options == 'Predictive Modeling':
    predictive_modeling()
elif options == 'Interactive Plots':
    team_data(df1)
    player_data(df2)