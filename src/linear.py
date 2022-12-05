import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")

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
    st.title("Streamlit Demo")
# -- We use the first column here as a dummy to add a space to the left

# -- Get the user input
year_col, team_col, log_x_col = st.columns([5, 5, 5])
with year_col:
    year_choice = st.slider(
        "What year would you like to examine?",
        min_value=1995,
        max_value=2020,
        step=1,
        value=2020,
    )
with team_col:
    Team_choice = st.selectbox(
        "What team would you like to look at?",
        ("All", "Suns", "Spurs", "Knicks", "Heat", "Lakers"),
    )
with log_x_col:
    log_x_choice = st.checkbox("Log X Axis?")

# -- Read in the data
# df = px.data.gapminder()
# -- Apply the year filter given by the user
filtered_df = df[(df.Team == Team_choice)]
# -- Apply the continent filter
if Team_choice != "All":
    filtered_df = filtered_df[filtered_df.Team == Team_choice]

# -- Create the figure in Plotly
fig = px.scatter(
    filtered_df,
    x="YEAR",
    y="%PTS\n3PT",
    size="%PTS\n2PT MR",
    color="Team",
    # hover_name="country",
    log_x=log_x_choice,
    size_max=60,
)
fig.update_layout(title="NBA Team's % of points from Mid Range and Three Point From 2000 to 2020 ")
# -- Input the Plotly chart to the Streamlit interface
st.plotly_chart(fig, use_container_width=True)