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
def home(): #uploaded_file
    # if uploaded_file:
    #     st.header('Begin exploring the data using the menu on the left')
    # else:
    #     st.header('To begin please upload a file')
    # Set page header
    st.title("Welcome to My Streamlit App!")


# Add instructions
    st.write("Before you get started, look at these steps:")
    st.write("1. Upload your data file(s) on the left!")
    st.write("2. If you want to see your dataset select Data Summary")
    st.write("3. If you are ready to research select interactive plots!")

# Add image
    st.image("../img/homepage.png", caption="What will you explore...?", use_column_width=True)
    

def data_summary(df1,df2):
    st.header('Statistics of Dataframe')
    st.write(df1)
    st.write(df2)
    # st.write(df.describe())


def team_data(df1):
    # Create two columns
    col1, col2 = st.columns(2)

    # Select one or more teams from dropdown list
    sorted_unique_team = sorted(df1.TEAM.unique())
    team_option = ['All Teams'] + sorted_unique_team
    selected_teams = st.multiselect('Team', team_option, default='All Teams')

    try:
        # Select appropriate data based on selected team(s)
        if not selected_teams:
            raise ValueError('Please select at least one team.')
        elif 'All Teams' in selected_teams:
            df_selected_teams = df1.copy()
        else:
            df_selected_teams = df1[df1.TEAM.isin(selected_teams)].copy()
    except ValueError as e:
        st.warning(str(e))
        return

    # Select X and Y axes for scatter plot
    exclude_cols = ['TEAM']

    x_axis_options = [col for col in df_selected_teams.columns if col not in exclude_cols]
    x_axis_val = col1.selectbox('Select the X-axis', options=x_axis_options)
    y_axis_options = [col for col in df_selected_teams.columns if col not in exclude_cols]
    y_axis_val = col2.selectbox('Select the Y-axis', options=y_axis_options)

    # Create scatter plot with linear regression trendline
    plot = px.scatter(df_selected_teams, x=x_axis_val, y=y_axis_val, color=df_selected_teams.TEAM, trendline='ols',
                      trendline_color_override='green', hover_name="TEAM", hover_data=["YEAR", "W"])

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

    # Display scatter plot
    st.plotly_chart(plot, use_container_width=True)



def player_data(df2):
    # Create two columns
    col3, col4, col5 = st.columns(3)
    
    # Select one or more teams from dropdown list
    sorted_unique_team = sorted(df2.TEAM.unique())
    teams_option = ['All Teams'] + sorted_unique_team
    selected_teams = col5.multiselect('Team', teams_option, default='All Teams')

    try:
        # Select appropriate data based on selected team(s)
        if not selected_teams:
            raise ValueError('Please select at least one team.')
        elif 'All Teams' in selected_teams:
            df_selected_teams = df2.copy()
        else:
            df_selected_teams = df2[df2.TEAM.isin(selected_teams)].copy()
    except ValueError as e:
        st.warning(str(e))
        return

    players = df_selected_teams['Player'].unique()
    selected_player = col5.selectbox("Select player", players)
    filtered_data_players = df_selected_teams[df_selected_teams['Player'] == selected_player]

    # Select X and Y axes for scatter plot
    # exlude unnecessary collumns (ones that contain stringified dates)
    exclude_cols = ['TEAM', 'Player']
    x_axis_options = [col for col in filtered_data_players.columns if col not in exclude_cols]
    y_axis_options = [col for col in filtered_data_players.columns if col not in exclude_cols]

    x_axis_val = col3.selectbox('Select the X-axis', options=x_axis_options, key="3")
    y_axis_val = col4.selectbox('Select the Y-axis', options=y_axis_options, key="4")

    

    plot = px.scatter(filtered_data_players, x=x_axis_val, y=y_axis_val, color='TEAM', trendline='ols',
                      trendline_color_override='green', hover_name="Player", hover_data=["TEAM", "YEAR"])

    # Add linear regression prediction for next season
    if x_axis_val == 'YEAR' and y_axis_val != 'YEAR':
        # Initialize a LinearRegression object
        lr = LinearRegression()
        # Select the data for X and y variables to fit the model
        X = filtered_data_players[filtered_data_players.YEAR < 2022][[x_axis_val]]
        y = filtered_data_players[filtered_data_players.YEAR < 2022][[y_axis_val]]
        # Fit the linear regression model using the X and y variables
        lr.fit(X, y)
        # Set the value of the next season to make a prediction for
        next_season = 2022
        # Reshape the next season value to be a 2D array for prediction
        next_year = np.array([next_season]).reshape(-1, 1)
        # Use the linear regression model to make a prediction for the next season
        next_stat = lr.predict(next_year)
        # Display the predicted value for the selected y-axis variable and next season
        st.write(f"Predicted {y_axis_val} for {next_season}: {next_stat[0][0]:.2f}")
        # Add a trace to the plot for the next season prediction
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
        # Compute and display evaluation metrics for the model
        # Evaluation metrics
        y_pred = lr.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean squared error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")
    
    # Display scatter plot
    st.plotly_chart(plot, use_container_width=True)


    
    



# Add a title and intro text
st.title('NBA Data Explorer')
st.text('This is a web app to allow exploration of NBA Data')

# Sidebar setup
st.sidebar.title('Sidebar')
upload_file1 = st.sidebar.file_uploader('Upload a file containing NBA data', key="1")
upload_file2 = st.sidebar.file_uploader('Upload a file containing NBA data', key ="2")
#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Summary', 'Interactive Plots'])

# Check if file has been uploaded
try:
    if upload_file1 is not None:
        df1 = pd.read_csv(upload_file1)
    if upload_file2 is not None:
        df2 = pd.read_csv(upload_file2)
except:
    print("No file was uploaded.")
    
# Navigation options
if options == 'Home':
    home()
elif options == 'Data Summary':
    data_summary(df1,df2)
elif options == 'Interactive Plots':
    team_data(df1)
    player_data(df2)