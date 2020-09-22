# **********************************
# **********MISSING VALUES**********
# **********************************

# Many libraries will throw errors in case of missing values, so data should be prepared.
# Drop columns (simply remove column) # Imputation (replacing missing values with predicted similar ones)
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
X_full = pd.read_csv('../input/train.csv', index_col='Id') X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

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










# *****************************************
# **********CATEGORICAL VARIABLES**********
# *****************************************

# Transform strings into numbers or something

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
# .dtypes == list of columns; (...) === create generator for tuple with columns from X_train which are 'object'

# --------------------Drop Categorical Variables--------------------
# Easiest and the worst one

drop_X_train = X_train.select_dtypes(exclude=['object'])

# --------------------Label Encoding--------------------
# Transform text into numbers. Works well with ordinal data, e.g time variables, or any measurements.
# Breakfast   Breakfast
# Every day   3
# Never       0
# Rarely      1

from sklearn.preprocessing import LabelEncoder
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
# Example of generating random values. Ideally it should be done manually, as numbers' values matter

# --------------------One-Hot Encoding--------------------
# Create new column for each possible categorical variable, and then insert appropriate number
# Color    Red  Yellow  Green
# Red      1    0       0
# Red      1    0       0
# Yellow   0    1       0
# Green    0    0       1


# --------------------------------------------------------------------------------
# Label Encoding and One-Hot Encoding in action 
# --------------------------------------------------------------------------------

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # handle_unknown - unknown values (for which was not created column), that may encounter in validation data; sparse=False - encoded columns are returned as numpy array, instead of spare matrix
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Drop and then add back encoded columns
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# *************EXAMPLE***************
# *************EXAMPLE***************
# *************EXAMPLE***************

# Data preparation
import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv('../input/train.csv', index_col='Id')
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Label encoding example

X_train['Condition2'].unique() !== X_valid['Condition2'].unique()
# There's a problem. Train and valid data frames may have different labels.

# train Condition2:    valid Condition2:
# aaa                  bbb
# bbb                  ddd
# aaa                  xxx
# ccc                  aaa

# Simplest solution - drop "ugly" columns, and encode "good" ones
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"] # All object cols
good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])] # Cols which has exact same sets of categorized values 
bad_label_cols = list(set(object_cols)-set(good_label_cols)) # Remove from all calls good one, and leave only bad

from sklearn.preprocessing import LabelEncoder
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])


# Cardinality
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols)) # Number of unique entries in each column
d = dict(zip(object_cols, object_nunique)) # Mapped to column name

# The more unique entries column has, the more columns for each entry added. This property called cardinality. 
# Low unique values = low cardinality, and vice versa
# E.g If dataset contains 10000 entries, and one column has 100 unique entries, we need to add 100 columns to each entry, with total of 990000 new entries (and dropping old column)

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10] # list of cols with low cardinality
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

reduced_X_train = X_train.drop(object_cols, axis=1);
reduced_X_valid = X_valid.drop(object_cols, axis=1);

OH_X_train = pd.concat([reduced_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([reduced_X_valid, OH_cols_valid], axis=1)


# Label Encoding + One-Hot Encoding

# Instead of dropping columns with high cardinality
# Apply label encoding on them

# 1. Separate cols by low & high cardinality
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

# 2. Check cols aren't dirty (have same sets of unique values in columns). Drop'em if they're.
good_label_cols = [col for col in high_cardinality_cols if set(X_train[col]) == set(X_valid[col]) == set(X_test[col])]
bad_label_cols = list(set(high_cardinality_cols)-set(good_label_cols))

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
label_X_test = X_test.drop(bad_label_cols, axis=1)

# 3. Create label encoder and fit it.
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
    label_X_test[col] = label_encoder.transform(X_test[col])

# 4. Create OneHot Encoder and fit and transform all other low cardinality columns.
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(label_X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(label_X_valid[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(label_X_test[low_cardinality_cols]))

OH_cols_train.index = label_X_train.index
OH_cols_valid.index = label_X_valid.index
OH_cols_test.index = label_X_test.index

reduced_X_train = label_X_train.drop(low_cardinality_cols, axis=1)
reduced_X_valid = label_X_valid.drop(low_cardinality_cols, axis=1)
reduced_X_test = label_X_test.drop(low_cardinality_cols, axis=1)

final_X_train = pd.concat([reduced_X_train, OH_cols_train], axis=1)
final_X_valid = pd.concat([reduced_X_valid, OH_cols_valid], axis=1)
final_X_test = pd.concat([reduced_X_test, OH_cols_test], axis=1)










# *****************************
# **********PIPELINES**********
# *****************************

# Pipelines are made for preprocessing data automation. It's less error prone, and easier to reuse.

# 1. Define steps
# a) Impute missing values in numerical data
# b) Impute values and apply a one-hot encoding to categorical data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 2. Define the model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

# 3. Use. Define pipeline and use it just as a regular model

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)









# ********************************************
# *************CROSS VALIDATION***************
# ********************************************

# Given limited data set. Taking first 20% for data validation may differ from another 20% and so on.
# Cross validation is the modeling process on different subsets to get multiple measure quality. Perfect for small datasets. As big datasets may have enough data for 20% data and it may process too slow.
# For example: dividing data into 5 pieces, each 20% of the full dataset - data is broken into 5 "folds"

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')











# ***********************************
# *************XGBoost***************
# ***********************************

# Random Forest Regressor is 'ensemble method'. Ensemble method combines predictions of several models (in case of RF - several trees).

# XGBoost is another ensemble method. XGBoost build decision tree one each time. Each new tree corrects errors which were made by previously trained decision tree.
# How it works?
# 1. Create first naive model
# 2. Make predictions
# 3. Calculate error
# 4. Use error to fit new model that will be added to ensemble
# 5. Repeat from #2

# RF takes only 2 params: n_estimators - amount of trees, and number of features to be seleceted.
# XGBoost takes lots more.

# 1. n_estimators - how many times cycle will be ran
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

# 2. early_stopping_rounds - says the model when to stop iterating, when validation score stops improving. Also by specifying early_stopping_rounds we need to include vaildation data - as model should compare predictions with real y.
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# 3. learning_rate - specifing small number will make each run help us less - but with that we can use larger n_estimators and avoid overfitting by that
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

# 4. n_jobs - multithreading, best to set to amount of cores of processor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)









# ****************************************
# *************DATA LEAKAGE***************
# ****************************************

# Data leakage is when you have strong predictions on train data, with very poor predictions on the prod data.

# Target leakage
# For example having strong connections between some columns 
# And in prod "real-time" data may occur "lag", when first dependent column is changed, but another is not. Such dependencies should be removed

# Train-test contamination
# Using test data in train data, for example impute values from fitted by valid data, not train.




