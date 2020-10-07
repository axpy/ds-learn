import pandas as pd

# DataFrame
pd.DataFrame({'Col A': [1,2,3], 'Col B': [2,3,4]}) # Default indexed 0, 1...
pd.DataFrame({'Col A': [1,2,3], 'Col B': [2,3,4]}, index=['Product A', 'Product B', 'Product C']) # Explicit indexes 



# Series (DataFrame column)
pd.Series([1,2,3,4])
pd.Series([1,2,3], index=['Product A', 'Product B', 'Product C'], name='Col A')



# CSV (Comma Separated Values)
# Product A,Product B,Product C,
# 30,21,9,
# 35,34,1,
# 41,11,11

pd.read_csv('filepath', index_col="index-col") # Return DataFrame with index_col (if presented)



# Accessing columns and values
df.some_column # or df['some column if there's a whitespace'] 
df.some_column[row_number] # accessing value by row num



# Index based selection (reversed, instead of column-row, there's row-column access)
df.iloc[0] # Get first row
df.iloc[0, 0] # Get first row and first column data
df.iloc[:, 0] # Get all rows (:), and their first column data
df.iloc[[1,2,3], 0] # Get rows 1,2,3, and their first column data

# More python list accessors: 
# [:3] - first three items (or 0:3) 0,1,2
# [1:3] - from 1 to 3 (excluding last) 1,2
# [-5:] - last 5 items



# Label based selection
# Same as index based, but:
# 1. Parameters are now matching index label, not just an index like array. 
# 2. Also range is inclusive
df.loc['Apples':'Potatoes', ['col1', 'col2']] # Get col1 col2 data for all entries from 'Apples' to 'Potatoes'

# Manipulating the index
df.set_index("another col for index")



# Conditional selection
df['some column'] == 'some value' # Series of whole DataFrame with True/False result for condition
df.loc[df['some solumn'] == 'some value'] # DataFrame with rows only passing the condition

reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)] 
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]

reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]



# Summary functions

# Display count, min, max etc
df['columnnmae'].describe()
df['columnnmae'].mean() # Or just by specific method
df['columnnmae'].unique() # List of unique values
df['columnnmae'].value_counts() # How often each value occurs

# Maps

# Remean points column to 0 
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

# Equivalent form with apply()
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns') # With axis='index', remean_points will recieve columns to be iterated

# Another way 
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean # New column with new mean

reviews.point/reviews.price # Create new column with point/price ratio

# Map and apply are pure immutable functions, they return new Series and DataFrames



# Idxmax/Idxmin
df.column.idxmax() # Get max value row index for that column
df.column.idxmin() # Get min value row index for that column
df.column.max() # Get max value of that column
df.column.min() # Get min value of that column


# Get title of best points/price ratio wine
wine_index = (reviews.points/reviews.price).idxmax()
bargain_wine = reviews.loc[wine_index, 'title']



# Total count of some word in description column
n_trop = reviews.description.map(lambda d: "tropical" in d).sum()
n_fruit = reviews.description.map(lambda d: "fruity" in d).sum()
descriptor_counts = pd.Series([n_trop, n_fruit], index=['tropical', 'fruity'])

# Output:
# tropical    3607
# fruity      9090
# dtype: int64




# Grouping 
# groupby - create groups (smaller DataFrames) where key is specified column name
# df.groupby() returns DataFrame-like object, from which we can take column, apply stuff

df.groupby('column').column.count() # How often each value occurs  
# same as df['columnnmae'].value_counts() 
# same as df.groupby('column').size()

df.groupby('points').price.min() # Get min price of each group, grouped by points
df.groupby('winery').apply(lambda df: df.title.iloc[-1]) # Get title of last item of each winery

df.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]) # Get Best pointed wines from each country-province
# Returns DataFrame with multiinedx, to bring back - use reset_index()

df.groupby('country').price.agg([len, min, max]) # agg - return list of Series with specified props

# Sorting

df.sort_values(by='column', acending=False)
df.sort_values(by=['or', 'multiple'])
df.sort_index()



# Data types

df.column.dtype # Return type of column
df.dtype # Return Series with col-type values
df.column.astype('some new type') # Return series as some other type

pd.isnull(df.column) # Return list of indexes of entries with null value in column
df[pd.notnull(df.column)] # Return new dataframe, filtered by columns with all not null values

df.column.fillna("Unknown") # Replace by word
df.column.replace('any value', 'by another any values')

reviews_per_region = reviews.region_1.fillna("Unknown").value_counts().sort_values(ascending=False)
# Replace all NaN values for region_1, count each, sort


# Rename

df.rename(columns={'a': 'A'})
df.rename(index={0: 'firstEntry', 1: 'secondEntry'})

df.rename_axis("name for row/entry", axis='rows').rename_axis("name for column, e.g - property", axis='columns')

# Combining

pd.concat([df_1, df_2]) # Combine two  data frames with same columns

# Joining
# Given 2 data frames. Find same events occured on the same day:

left = df_1.set_index(['title', 'date']) # As title is unique, date may have multiple events
right = df_2.set_index(['title', 'date']) # Same for second DF

left.join(right, lsuffix='_CAN', rsuffix='_UK') # Join two DFs, with suffixed columns (as they have same names)

