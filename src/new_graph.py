import streamlit as st 
import pandas as pd 
import numpy as np 
import pydeck as pdk 
import altair as alt 
from datetime import datetime

DATA_URL = ('covid.csv')
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%DD-%MM-%YYYY')
    return data
df = load_data()

st.write(df)


# Filters UI
subset_data = df
country_name_input = st.sidebar.multiselect(
'Country name',
df.groupby('Country').count().reset_index()['Country'].tolist())
# by country name
if len(country_name_input) > 0:
    subset_data = df[df['Country'].isin(country_name_input)]



metrics =['total_cases','new_cases','total_deaths','new_deaths','total_cases_per_million','new_cases_per_million','total_deaths_per_million','new_deaths_per_million','total_tests','new_tests','total_tests_per_thousand','new_tests_per_thousand']
cols = st.selectbox('Covid metric to view', metrics)
# let's ask the user which column should be used as Index
if cols in metrics:   
    metric_to_show_in_covid_Layer = cols


## linechart

st.subheader('Comparision of infection growth')
total_cases_graph  =alt.Chart(subset_data).transform_filter(
   alt.datum.total_cases > 0  
).mark_line().encode(
    x=alt.X('Date', type='nominal', title='Date'),
    y=alt.Y('sum(total_cases):Q',  title='Confirmed cases'),
    color='Country',
    tooltip = 'sum(total_cases)',
).properties(
    width=1500,
    height=600
).configure_axis(
    labelFontSize=17,
    titleFontSize=20
)

st.altair_chart(total_cases_graph)