# **********INTRO**********

# pandas as pd - library for data reading;
# e.g:
home_data = pd.read_csv(some_data_file_pah) # DataFrame from csv
home_data.describe() # Display summary statistics of data
home_data.columns # Columns of data frame
home_data.dropna(axis=0) # Drop not available (na) rows (axis=0). Not available - when some entries has missing values. dropna(axis=1) - drop columns, which doesn't have exactly all data.

y = home_data.Price # Prediction target, by convention, named 'y'. DataFrame with single `Price` column
X = home_data['Rooms', 'Area', ....] # Features, which are inputted in model (and later used to make predictions). DataFrame with specified columns
X.describe() # Statistics with only featured columns
X.head() # Quick overview of 5 top entries of data

# scikit-learn - library for creating models
from sklearn import DecisionTreeRegressor

home_model = DecisionTreeRegressor(random_state=1) 
# DecisionTree ml model. Depending on columns, each node asks a question with only True/False answer. The final termination node (or leaf) gives prediction. 
# Random state. Every training allows some randomness, so for having same result on every run, we need to specify the random state.


# ````````````In-Sample`````````````
home_model.fit(X, y) # Create patterns from features with data and prediction target.

prediction = home_model.predict(X.head())
# Predict Prices for top 5 entries of data. Same DataFrame as `y`, but with predicted prices.

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y, prediction) # Difference between prediction and actual data. It'll be quite small here, as it has `in-sample` scores - to be predicted entries were already fitted model 


# ````````````Out-Sample`````````````
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Exclude some data from model-building process, and then use it for testing model's accuracy. This data called `validation data` 

model = DecisionTreeRegressor() 
model.fit(train_X, train_y)
prediction = model.predict(val_X)
mean_absolute_error(val_y, prediction) # Much higher.

# ----------------------------------------------------------------------------------------------------
# The steps to building and using a model are:
# 
# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.
# ----------------------------------------------------------------------------------------------------

# **********UNDER/OVERLIFTING**********

# Underlifting vs Overlifting

# Each node may have two or 1000 nodes. Having litle nodes will end up with a small amount of groups. Too many nodes will affect on huge amount of groups and very sensitive to some parameters, that actually are not.
# Truth is somewhere between.

# RandomForestRegressor - is model which has many trees with different max_nodes_leaf numbers, and then making predictions by averaging predictions from those trees








