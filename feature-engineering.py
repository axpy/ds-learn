# Prep data

# Remove rows by condition
df = df.query('state != "live"') # Remove all rows, where state is live
df = df[df.state != 'live'] # Same as above



# Add new column by data from other
df.assign(column_name=(df['another_column'] == 'smth').astype(int)) # Create new column with 0 or 1 value. 1 if another_column's value == 'smth', 0 if not



# Convert dates to separate columns
df = df.assign(hour=df.time_col.dt.hour,
               day=df.time_col.dt.day,
               month=df.time_col.dt.month,
               year=df.time_col.dt.year)
# Timestampt datatypes have special property .dt, with all needed data



# Label encode categorical values
le = LabelEncoder()
features_to_encode = ['abc', 'asd', 'zxc']

encoded = df[features_to_encode].apply(le.fit_transform)

final_data = df[['needed', 'cols', 'only']].join(encoded) # Add the needed columns with encoded ones (instead of remove and add)



# Data split by hands
# Ex: 10% validation, 10% testing, 80% training
valid_fraction = 0.1
valid_size = int(len(data) * valid_fraction)

train = data[:-2 * valid_size]
valid = data[-2 * valid_size:-valid_size]
test = data[-valid_size:]


# Light GBM for train
import lightgbm as lgb

feature_cols = train.columns.drop('outcome')

dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

# And metrics
from sklearn import metrics
ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['outcome'], ypred)

print(f"Test AUC score: {score}")


##################
# MORE ENCODINGS #
# MORE ENCODINGS #
# MORE ENCODINGS #
##################

# Count encoding
# Instead of random/hash categorical data replacements - replace with number of appearences in the column. E.g 10 occured 10 times - replace by "10"
import category_encoders as ce
cat_features = ['category', 'currency', 'country']

# Create the encoder
count_enc = ce.CountEncoder(cols=cat_features)

# Transform the features, rename the columns with the _count suffix, and join to dataframe
count_encoded = count_enc.fit_transform(ks[cat_features])
data = data.join(count_encoded.add_suffix("_count"))




# Target encoding
# Replace with average Y value for that categorical value. E.g entries with country = 'CA' has target value in average 0.28. Then replace 'CA' by '0.28'
# Create the encoder
target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])

# Transform the features, rename the columns with _target suffix, and join to dataframe
train_TE = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_TE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))





# CatBoost Encoding
# Same as Target encoding, but the average is taken from rows that was already checked. So for example first row will always have 1, second, if it's not the same as first - 0.5. and so on
# Create the encoder
target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])

# Transform the features, rename columns with _cb suffix, and join to dataframe
train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_CBE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))








######################
# FEATURE GENERATION #
######################

# Interactions (combining categorical variables
interactions = df['some category'] + '_' + df['another one'] # New Series combining values from two columns
le = LabelEncoder()
final_data = df.assign('some_category_another_one'=le.fit_transform(interactions))


# Features from date
# Number of projects launched in the last week
# Create Series with launched timestamp as index, and it's value - df's index. Like inverse view
launched = pd.Series(df.index, index=df.launched, name="count_7_days").sort_index()

# Launched - index: date, value: ip (index from df)
#1970-01-01 01:00:00    945713
#1970-01-01 01:00:00    319002
#1970-01-01 01:00:00    247913


# Next - use .rolling(time_range) method. Applying .rolling on Series with timestamp as index will return new Series with same index and value - amount of entries for that range, beginning from index
count_7_days = launched.rolling('7d').count() - 1

# Count_7_days - index: date, value: last 7 days count
#2009-04-21 21:02:48     0.0
#2009-04-23 00:07:53     1.0
#2009-04-24 21:52:03     2.0


# Reindex and later join
count_7_days.index = launched.values
count_7_days = count_7_days.reindex(df.index)
df.join(count_7_days)



# Time since the last project in the same category

def time_since_last_project(series):
    # diff() - get difference from previous value. e.g first value == NaN, as nothing to compare. second = first - second
    return series.diff().dt.total_seconds() / 3600.

df = ks[['category', 'launched']].sort_values('launched')

# git differences only by groups
timedeltas = df.groupby('category').transform(time_since_last_project)

#	launched
#94579 	NaN
#319002 NaN
#247913	0.000
#48147 	13.3
#75397 	NaN

# Fill NaN by medians. Reindex Series
timedeltas = timedeltas.fillna(timedeltas.median()).reindex(baseline_data.index)

# Transforming numeric features
# With huge distribution some models may have problems. E.g 95% are < 20k, and 5% may raise up to 100k. Then it may be helpful to normalize the distribution. Two approaches: log them or square root.

np.sqrt(df.column_with_bad_distribution)
np.log(df.column_with_bad_distribution)






# Combine multiple columns

import itertools

cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)
le = preprocessing.LabelEncoder()

# itertools.combinations returns array [A,B] where A and B are all possible combinations of cat_features
for f1, f2 in itertools.combinations(cat_features, 2):
    new_feature = clicks[f1].map(str) + '_' + clicks[f2].map(str)
    interactions[f1 + '_' + f2] = le.fit_transform(new_feature)



# Function for group transformation creating last total of 6h
def count_past_events(series):
    print(series)
    count_series = pd.Series(series.index, index=series.values)
    return count_series.rolling('6h').count() - 1


# Function for calculating all previous occurances and without actual's entry result (by - series)
def previous_attributions(series):
    """Returns a series with the number of times an app has been downloaded."""
    return series.expanding(2).sum() - series



#####################
# FEATURE SELECTION #
#####################

# As amount of features grow, the case of overfiting or just simply useless features may grow.
# There's a need in removing unnecessary ones.



from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = baseline_data.columns.drop('outcome')

# Keep 5 features
selector = SelectKBest(f_classif, k=5)

# Find best 5 most correlative features
X_new = selector.fit_transform(baseline_data[feature_cols], baseline_data['outcome'])

# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train.index, 
                                 columns=feature_cols)


#
# 	goal 	hour 	day 	month 	year 	category 	currency 	country 	category_currency 	category_country 	currency_country 	count_7_days 	time_since_last_project
#0 	0.0 	0.0 	0.0 	0.0 	2015.0 	0.0 	5.0 	9.0 	0.0 	0.0 	18.0 	1409.0 	0.0
#1 	0.0 	0.0 	0.0 	0.0 	2017.0 	0.0 	13.0 	22.0 	0.0 	0.0 	31.0 	957.0 	0.0
#2 	0.0 	0.0 	0.0 	0.0 	2013.0 	0.0 	13.0 	22.0 	0.0 	0.0 	31.0 	739.0 	0.0
#3 	0.0 	0.0 	0.0 	0.0 	2012.0 	0.0 	13.0 	22.0 	0.0 	0.0 	31.0 	907.0 	0.0
#4 	0.0 	0.0 	0.0 	0.0 	2015.0 	0.0 	13.0 	22.0 	0.0 	0.0
#


dropped_colunms = selected_features.columns[selected_features.var() == 0]



# L1 regularization - apply penalty by each features as complexity grow. For some columns penalty won't affect - those are low relevant.





























