import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

def interactive_plot():
    #USE SCATTER PLOT
    col1, col2 = st.columns(2)
    
    sorted_unique_team = sorted(df1.TEAM.unique())
    # selected_team = st.multiselect('Team', sorted_unique_team, sorted_unique_team)
    # st.header("You selected: {}".format(", ".join(selected_team)))
    
    x_axis_val = col1.selectbox('Select the X-axis', options=df1.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df1.columns)

    plot = px.scatter(df1, x=x_axis_val, y=y_axis_val, color = df1.TEAM, trendline='ols',
                      trendline_color_override='green', hover_name = "TEAM", hover_data=["YEAR", "W"] )
    
    

    st.plotly_chart(plot, use_container_width=True)
    a = px.get_trendline_results(plot).px_fit_results.iloc[0].rsquared
    st.text("R-squared value: ")
    if a >= .5:
        st.markdown(a)
        st.text('\u2713')
    else:
        st.markdown(a)
        st.text('\u274c')



    col3, col4, col5 = st.columns(3)
    
    sorted_unique_team = sorted(df2.TEAM.unique())
    # selected_team = st.multiselect('Team', sorted_unique_team, sorted_unique_team)
    # st.header("You selected: {}".format(", ".join(selected_team)))
    

    players = df2['Player'].unique()
    selected_player = col5.selectbox("Select player", players)
    filtered_data_players = df2[df2['Player'] == selected_player]

    df2['TEAM'].unique()
    # selected_team = col6.selectbox("Select Team", teams)
    # filtered_data_team = df2[df2['TEAM'] == selected_team]


    x_axis_val = col3.selectbox('Select the X-axis', options=df2.columns, key = "3")
    y_axis_val = col4.selectbox('Select the Y-axis', options=df2.columns, key = "4")

    # took out df2 df2.TEAM
    plot = px.scatter(filtered_data_players , x=x_axis_val, y=y_axis_val, color = 'TEAM', trendline='ols',
                       trendline_color_override='green', hover_name = "Player", hover_data=["TEAM", "YEAR"])
    
    
    # text = 'YEAR',
    # plot.update_traces(textposition="bottom right")
    st.plotly_chart(plot, use_container_width=True)

    # results = px.get_trendline_results(plot)
    # st.text(results)

    # display r-square value, however only getting first index which is the suns
    a = px.get_trendline_results(plot).px_fit_results.iloc[0].rsquared
    st.text("R-squared value: ")
    if a >= .5:
        st.markdown(a)
        st.text('\u2713')
    else:
        st.markdown(a)
        st.text('\u274c')
    
    



# Add a title and intro text
st.title('NBA Data Explorer')
st.text('This is a web app to allow exploration of NBA Data')

