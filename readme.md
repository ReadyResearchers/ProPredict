# Linear Regression Analysis of Basketballs Top Offensive Strategies

**Background**
This sports analytics web app utilizes Streamlit's interactive dashboard to conduct research and analysis on NBA data, with a focus on changes in offensive strategies from the late 1990s to the end of the 2021 season. The app leverages Streamlit's powerful graphing tools, which enable users to select elements from the dataset and visualize correlations between statistics, facilitating the analysis of changes over time.



# Installation(s)

To install and run the web app, please follow these steps:

1. Install streamlit using pip:

```text
pip install streamlit
```

2. Clone the repository and navigate to the project directory:

```text
git clone https://github.com/username/repo-name.git
cd repo-name
```

3. Install the required packages using pip:

```text
pip install package_name
```

4. Run the app!

```text
streamlit run <name_of_python_file>
```

# Project Structure

`data` folder:

 - Contains the sports csv data set you would like to conduct research upon
 
NBA offensive team stats is located here as an example csv file

`src` folder:

 - Contains all source code located in `main.py` for the implementation of streamlit tool

# Usage

After all dependencies are installed you can run the following command : `streamlit run main.py`

- This will start the Streamlit app and open it in your default browser.
- If you encounter any errors or warnings, check the console output for details.

1. On the home page of the Streamlit app, on the left hand side-bar you will see a button that allows you to upload a CSV file.

- Click the "Browse files" button to select a CSV file from your computer.

2. Click the "Interactive Plots" button on the lower-left to analyze the selected dataset and display the results in a scatter plot

- The scatter plot shows the relationship between the X and Y elements, with each data point represented as a circle.
- The line of best fit  shows the slope, intercept, and R-squared value of the linear regression line.

3. Investigate and explore the data points by interacting with the graphs and the table.

- Use the zoom and pan controls on the scatter plot to focus on specific areas of the graph.
- Hover over a data point on the scatter plot to see its X and Y values.
- Click on a data point to see more detailed information in the table.

4. Repeat steps 3-5 to analyze other CSV files or different combinations of X and Y elements.

- The app supports uploading different CSV files and selecting different X and Y elements for each file.


5. Close the Streamlit app by pressing Ctrl+C in your terminal or closing the browser window.


# Results section: pending

Conclusion:

Summarizes the key challenges and results
Captures the attention of the reader
Convinces the reader to explore rest of senior thesis
References details and results whenever possible!
To get started, use the abstract as a “project roadmap”
