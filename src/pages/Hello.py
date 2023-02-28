import streamlit as st
import pandas as pd

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
st.set_page_config(layout="wide")

# Functions for each of the pages
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')