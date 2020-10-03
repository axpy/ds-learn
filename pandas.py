import pandas as pd

# DataFrame
pd.DataFrame({'Col A': [1,2,3], 'Col B': [2,3,4]}) # Default indexed 0, 1...
pd.DataFrame({'Col A': [1,2,3], 'Col B': [2,3,4]}, index=['Product A', 'Product B']) # Explicit indexes 

# Series (DataFrame column)
pd.Series([1,2,3,4])
pd.Series([1,2,3], index=['Product A', 'Product B'], name='Col A')

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

