import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np 
import pydeck as pdk 
import altair as alt 

st.set_page_config(layout="wide")


# def select_dataset():
uploaded_file = st.file_uploader("Please select your dataset")
df = pd.read_csv(uploaded_file)


st.write(df)
    



# -- Create three columns
col1, col2, col3 = st.columns([5, 5, 20])
# -- Put the image in the middle column
# - Commented out here so that the file will run without having the image downloaded
# with col2:
# st.image("streamlit.png", width=200)
# -- Put the title in the last column
with col3:
    st.title("Evolution of NBA Baketball")
# -- We use the first column here as a dummy to add a space to the left

# Filters Dataset
subset_data = df
team_name_input = st.sidebar.multiselect(
'Team name',
df.groupby('Team').count().reset_index()['Team'].tolist())
# by country name
if len(team_name_input) > 0:
    subset_data = df[df['Team'].isin(team_name_input)]

metrics =['YEAR','GP','MIN','"%FGA2PT"','Team']
cols = st.selectbox('Offensive statistic to view', metrics)
# let's ask the user which column should be used as Index
if cols in metrics:   
    metric_to_show = cols


## linechart

st.subheader('Comparision of Offensive Statistics')
GP_graph  =alt.Chart(subset_data).transform_filter(
   alt.datum.GP > 0  
).mark_line().encode(
    x=alt.X('Team', type='nominal', title='Team'),
    y=alt.Y('sum(GP):Q',  title='GP'),
    color='Team',
    tooltip = 'sum(GP)',
).properties(
    width=1500,
    height=600
).configure_axis(
    labelFontSize=17,
    titleFontSize=20
)

st.altair_chart(GP_graph)
