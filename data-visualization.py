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
