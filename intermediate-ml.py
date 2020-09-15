# **********MISSING VALUES**********

# Many libraries will throw errors in case of missing values, so data should be prepared.
# Drop columns (simply remove column)
# Imputation (replacing missing values with predicted similar ones)
# Extension to imputation (imputation + creating new boolean column, noting is value from that column was synthetically imputed)

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # List of columns with missing data

# -----------------Drop columns-----------------
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_train.drop(cols_with_missing, axis=1)

# -----------------Imputation-----------------
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train)) # Create new DataFrame; fit imputer with missing values and transform training dataset. Could be done with two separate steps
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid)) # imputer already has pattern for creating missing values replacements from X_train, then just simply transform

# Return back column names as imputation **For some reason replaced columns with numbers**
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# -----------------Extension to Imputation-----------------

X_train_copy = X_train.copy()
X_valid_copy = X_valid.copy()
# Make copies to not modify original

for col in cols_with_missing:
    X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()
    X_valid_copy[col + '_was_missing'] = X_valid_copy[col].isnull()

imputer = SimpleImputer()
imputed_X_train_copy = pd.DataFrame(imputer.fit_transform(X_train_copy))
imputed_X_valid_copy = pd.DataFrame(imputer.transform(X_valid_copy))

# Imputation removed column names; put them back
imputed_X_train_copy.columns = X_train_copy.columns
imputed_X_valid_copy.columns = X_valid_copy.columns

# **********EXAMPLE**********

import pandas as pd
from sklearn.model_selection import train_test_split
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True) # Remove rows with NaN SalesPrices values
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True) 

# Remove non-numeric columns
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print(X_train.shape) # (1168, 36) 1168 rows, 36 columns

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum()) # isnull() - return same DataFrame with NaN and similar values replaced by True, and all others with False; sum() - return `column - amount_of_missing_entries` 
print(missing_val_count_by_column[missing_val_count_by_column > 0]) # return colmns with at least one missing value

# LotFrontage    212
# MasVnrArea       6
# GarageYrBlt     58

# Dropping columns will drop 3 columns. For instance LotFrontage has 1/5 missing values, and somehow not bad idea. But for MasVnrArea - it's 0.05% of missing values - bad idea to drop column

# -----------------Drop columns-----------------
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # Your code here
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# -----------------Imputation-----------------
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer.fit(X_train)
imputed_X_train = pd.DataFrame(imputer.transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_train.columns

# -----------------Extension to Imputation-----------------

X_train_copy = X_train.copy()
X_valid_copy = X_valid.copy()

for col in cols_with_missing:
    X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()
    X_valid_copy[col + '_was_missing'] = X_valid_copy[col].isnull()

imputer = SimpleImputer()
final_X_train = pd.DataFrame(imputer.fit_transform(X_train_copy))
final_X_valid = pd.DataFrame(imputer.transform(X_valid_copy))

final_X_train.columns = X_train_copy.columns
final_X_valid.columns = X_valid_copy.columns


# -----------------Drop Columns + Extended Imputation-----------------

X_train_copy = X_train.copy()
X_valid_copy = X_valid.copy()

cols_with_missing = [col for col in X_test.columns if (X_train_copy[col].isnull().any() and col is not 'GarageYrBlt')]
cols_with_missing.remove('GarageYrBlt')

for col in cols_with_missing:
    X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()
    X_valid_copy[col + '_was_missing'] = X_valid_copy[col].isnull()

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

final_imputer = SimpleImputer(strategy='constant', fill_value=0)
final_X_train = pd.DataFrame(final_imputer.fit_transform(reduced_X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(reduced_X_valid))

final_X_train.columns = reduced_X_train.columns
final_X_valid.columns = reduced_X_train.columns

