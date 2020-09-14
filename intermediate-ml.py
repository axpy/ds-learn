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

# Return back column names as imputation removed them
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# -----------------Extension to Imputation-----------------

X_train_copy = X_train.copy()
X_valid_copy = X_valid.copy()
# Make copies to not modify original

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

# **********EXAMPLE**********

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

