#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

# Functions for each of the pages
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

def data_summary():
	st.header('Statistics of Dataframe')
	st.write(df1)
    # st.write(df.describe())

def data_header():
    st.header('Header of Dataframe')
    st.write(df1.head())

def displayplot():
    st.header('Plot of Data')
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=df1['TEAM'], y=df1['GP'])
    ax.set_xlabel('TEAM')
    ax.set_ylabel('GP')
    
    st.pyplot(fig)

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
    
    # plot.update_traces(mode="markers+lines", hovertemplate=None)
    # plot.update_layout(hovermode="x unified")
    # text = 'YEAR',
    # plot.update_traces(textposition="bottom right")
    st.plotly_chart(plot, use_container_width=True)
    a = px.get_trendline_results(plot).px_fit_results.iloc[0].rsquared
    st.text("R-squared value: ")
    if a >= .5:
        st.markdown(a)
        st.text('\u2713')
    else:
        st.markdown(a)
        st.text('\u274c')



    col3, col4 = st.columns(2)
    
    sorted_unique_team = sorted(df2.TEAM.unique())
    # selected_team = st.multiselect('Team', sorted_unique_team, sorted_unique_team)
    # st.header("You selected: {}".format(", ".join(selected_team)))
    
    x_axis_val = col3.selectbox('Select the X-axis', options=df2.columns, key = "3")
    y_axis_val = col4.selectbox('Select the Y-axis', options=df2.columns, key = "4")

    plot = px.scatter(df2, x=x_axis_val, y=y_axis_val, color = df2.TEAM, trendline='ols',
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
if upload_file2 is not None:
    df2 = pd.read_csv(upload_file2)
    # df = df.astype({'YEAR':'int'})
    # df.YEAR = pd.DatetimeIndex(df.YEAR).strftime("%Y")

# Navigation options
if options == 'Home':
    home(upload_file1)
elif options == 'Data Summary':
    data_summary()
elif options == 'Data Header':
    data_header()
elif options == 'Scatter Plot':
    displayplot()
elif options == 'Interactive Plots':
    interactive_plot()