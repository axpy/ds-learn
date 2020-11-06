from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

num_tables = len(list(client.list_tables(dataset))) # Amount of tables in dataset (list_tables returns iterator)

# List all the tables in the dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there is only one!)
for table in tables:  
    print(table.table_id)

# Or another way to get list of names of tables
list_of_tables = [table.table_id for table in tables]

# Get specific table
table_ref = dataset_ref.table('crime')
crime_table = client.get_table(table_ref)

# Get top 5 rows as data frame
df = client.list_rows(crime_table, max_results=5).to_dataframe()

# Get amount of columns with 'TIMESTAMP' type
num_timestamp_fields = sum([1 for x in crime_table.schema if 'TIMESTAMP' in x.field_type])





# Queries

# Set limit query size to 5MB
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=1000*1000*5)

# Or
# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)
dry_run_query_job.total_bytes_processed == 429036722  # This query will process 429036722 bytes.



# Select country value from global_air_quality table, where unit = 'ppm'
first_query = """
              SELECT country
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit = 'ppm'
              """

# API request - run the query, and return a pandas DataFrame
first_results = first_query_job.to_dataframe()



# Aggregation functions
# COUNT(1), SUM(), AVG(), MIN(), MAX()

# Aliases: 'AS' keyword + new name for column

# Group By
# "Divide" table into multiple sub-tables with the same value for specified column 

# WHERE - filter `SELECT` operation
# HAVING - filter `GROUP BY` operation

# E.g 'comments' table
# Get authors with > 10000 comments
prolific_commenters_query = """
        SELECT author, COUNT(1) AS NumPosts
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY author
        HAVING COUNT(1) > 10000
        """

#         author  NumPosts
#0  dragonwriter     10723
#1          None    227736
#2           eru     10448
#3       rbanffy     10557
#4         DanBC     12902

# Note about group and select
# Selections from Grouped table may be proceeded only with specified group colunm, or by aggregation functions.
# In examlpe above - group by author, and select exact same author column + count aggr
# You can't select e.g age or something else, not included in groups



# Dates
# Two date types: DATE and DATETIME
# EXTRACT function can extract some specific data from complex types:
# e.g EXTRACT(DAY from Date) AS Day - create column `Day` with given day from date

# Get most accidental day of week from accidents table
query = """
        SELECT COUNT(1) AS num_accidents,
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """

# Average spendings on SE.XPD.TOTL.GD.ZS between two years by country
country_spend_pct_query = """
                          SELECT country_name, AVG(value) AS avg_ed_spending_pct
                          FROM `bigquery-public-data.world_bank_intl_education.international_education`
                          WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS' and year >= 2010 and year <= 2017
                          GROUP BY country_name
                          ORDER BY avg_ed_spending_pct DESC
                          """

# FULL CYCLE QUERY #
# FULL CYCLE QUERY #
# FULL CYCLE QUERY #
# FULL CYCLE QUERY #

rides_per_year_query = """
                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) AS num_trips
                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                       GROUP BY year
                       ORDER BY num_trips DESC
                       """

# Same but per month and with 2017 year
rides_per_month_query = """
                        SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month, COUNT(1) AS num_trips
                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017
                        GROUP BY month
                        ORDER BY num_trips DESC
                        """

# Set up the query (cancel the query if it would use too much of 
# your quota)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**7)
rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
rides_per_year_result = rides_per_year_query_job.to_dataframe() # Your code goes here

# FULL CYCLE QUERY #
# FULL CYCLE QUERY #
# FULL CYCLE QUERY #
# FULL CYCLE QUERY #


# WITH ... AS
# Used for code extracting and make it more readable

# First get whole table and filter it, then use already filtered data
speeds_query = """
               WITH RelevantRides AS
               (
                   SELECT
                       EXTRACT(HOUR from trip_start_timestamp) AS hour_of_day,
                       trip_miles,
                       trip_seconds
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE
                       trip_start_timestamp > '2017-01-01' AND
                       trip_start_timestamp < '2017-07-01' AND
                       trip_seconds > 0 AND
                       trip_miles > 0
               )
               SELECT hour_of_day, COUNT(1) AS num_trips, (3600 * SUM(trip_miles) / SUM(trip_seconds)) AS avg_mph
               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """

# JOINS
# Inner Join: find only matches between two tables
# 
# SELECT *
# FROM a
# INNER JOIN b ON a.id = q.parent_id
#
# 1. Create new table with all columns from `a` and `b`, labeled by `a.column_name` and `b.column_name`
# 2. Iterate over rows of `a` and find matches with `b` rows from `ON` condition
# 3. With multiple matches - add multiple items



# EXAMPLE
# 1. Create new table with rows containing answers and questions (there will be lots of duplicates of question data, but with different answers)
# 2. Filter them by `q.tags` containing *bigquery* word
# 3. Group by user id, so it'll look like bunches of answers for each user
# 4. Sort by total amount of answers
bigquery_experts_query = """
                         SELECT a.owner_user_id AS user_id, COUNT(1) as number_of_answers
                         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                         INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                             ON q.id = a.parent_id
                         WHERE q.tags LIKE '%bigquery%'
                         GROUP BY user_id
                         ORDER BY number_of_answers DESC
                         """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
bigquery_experts_query_job = client.query(bigquery_experts_query, job_config=safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
bigquery_experts_results = bigquery_experts_query_job.to_dataframe() # Your code goes here





























































