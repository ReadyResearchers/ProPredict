#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime

st.set_page_config(layout="wide")

# Functions for each of the pages
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

def data_summary():
	st.header('Statistics of Dataframe')
	st.write(df)
    # st.write(df.describe())

def data_header():
    st.header('Header of Dataframe')
    st.write(df.head())

def displayplot():
    st.header('Plot of Data')
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=df['Team'], y=df['GP'])
    ax.set_xlabel('Team')
    ax.set_ylabel('GP')
    
    st.pyplot(fig)

def interactive_plot():
    #USE SCATTER PLOT
    col1, col2 = st.columns(2)
    
    sorted_unique_team = sorted(df.Team.unique())
    selected_team = st.multiselect('Team', sorted_unique_team, sorted_unique_team)
    st.header("You selected: {}".format(", ".join(selected_team)))
    

    x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val, color = "Team", trendline='ols', trendline_color_override='darkblue')
    st.plotly_chart(plot, use_container_width=True)

# Add a title and intro text
st.title('NBA Data Explorer')
st.text('This is a web app to allow exploration of NBA Data')

# Sidebar setup
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Upload a file containing NBA data')
#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Summary', 'Data Header', 'Scatter Plot', 'Interactive Plots'])

# Check if file has been uploaded
if upload_file is not None:
    df = pd.read_csv(upload_file)
    df['YEAR'] = pd.to_datetime(df['YEAR'])

# Navigation options
if options == 'Home':
    home(upload_file)
elif options == 'Data Summary':
    data_summary()
elif options == 'Data Header':
    data_header()
elif options == 'Scatter Plot':
    displayplot()
elif options == 'Interactive Plots':
    interactive_plot()