# Add linear regression prediction for next season
    if x_axis_val == 'YEAR' and y_axis_val != 'YEAR':
        # Initialize a LinearRegression object
        lr = LinearRegression()
        # Select the data for X and y variables to fit the model
        X = df_selected_teams[df_selected_teams.YEAR < 2022][[x_axis_val]]
        y = df_selected_teams[df_selected_teams.YEAR < 2022][[y_axis_val]]
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