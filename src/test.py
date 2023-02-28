import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load sports dataset into a pandas DataFrame
df = pd.read_csv('/Users/liam/Desktop/CS600/Evolution-of-NBA-Basketball/data/players.csv')

# Create a list of player names
player_names = df['Player'].unique()

# Create dropdown menu
dropdown = go.layout.Dropdown(
    options=[{'label': name, 'value': name} for name in player_names],
    value=player_names[0],
)

# Define function to filter data for selected player
def filter_data(selected_player):
    return df[df['Player'] == selected_player]

# Create initial scatter plot
fig = px.scatter(df, x='Player', y='PTS', color='player_name')

# Add dropdown menu to plot
fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            buttons=[
                go.layout.button(
                    label=name,
                    method='update',
                    args=[{'x': [filter_data(name)['x_data']],
                           'y': [filter_data(name)['y_data']],
                           'marker.color': [filter_data(name)['player_name']]}],
                )
                for name in player_names
            ],
            direction='down',
            showactive=True,
            active=0,
            x=0.1,
            y=1.1
        )
    ]
)

# Display the scatter plot
fig.show()
