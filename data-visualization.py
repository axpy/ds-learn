import matplotlib.pyplot as plt # Library for data plotting
import seaborn as sns # Charts library - eating plotted data
import pandas as pd

some_data = pd.read_csv(some_data_csv_file_path, index_col="Some index col, like `Date`", parse_dates=True)

# Line chart
plt.figure(figsize=(14,6))
plt.title("Some title")
plt.xlabel("Some x label")

# Plot whole data
sns.lineplot(data=some_data) 
# Or only specific columns
sns.lineplot(data=soe_data['Specific column'], label='Specific label')
sns.lineplot(data=soe_data['Specific column2'], label='Specific label2')


# Bar chart
sns.barplot(x=some_data.index, y=some_data['SOME_COLUMN'])

# Heat map
sns.heatmap(data=some_data, annot=True)

ign_data.loc['PC'].sort_values() # Get row 'PC' and sort it's values

# Scatter plot
sns.scatterplot(x=some_data['col-1'], y=some_data['col-2'])
sns.regplot(x=some_data['col-1'], y=some_data['col-2']) # Add regression line (the more steeper, the bigger correlation)

sns.scatterplot(x=some_data['col-1'], y=some_data['col-2'], hue=some_data['col-3']) # '3rd' axis as a color indicator

sns.lmplot(x="bmi", y="charges", hue="smoker", data=some_data) # The sum of two regressions

sns.swarmplot(x=some_data['some binary-like data column'], y=some_data['col-2']) # Categorical scatter plot


# Histogram

sns.displot(a=some_data['Some column'], kde=False) # Very similar to bar chart (kde is kernel density estimation, smoothing and esimation random variable accoring to other inputs)

# Density plot

sns.kdeplot(data=some_data['Some column'], shade=True) # Smoothed histogram (shade - colored area under the curve)

# 2D KDE plot

sns.jointplot(x=some_data['col1'], y=some_data['col2'], kind='kde')

# Multiple histogram
sns.distplot(a=some_data_1['col1'], label='col1\'s label', kde=False)
sns.distplot(a=some_data_2['col1'], label='col1\'s label', kde=False)
sns.distplot(a=some_data_3['col1'], label='col1\'s label', kde=False)

# Multiple kdes
sns.kdeplot(data=some_data_1['col1'], label='col1\'s label', shade=True)
sns.kdeplot(data=some_data_2['col1'], label='col1\'s label', shade=True)
sns.kdeplot(data=some_data_3['col1'], label='col1\'s label', shade=True)

plt.legend() # As for multiple plots, legend should be expliticly set


