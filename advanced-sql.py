# Inner join
# Get matches only from both sides
# 
# Left/Right join
# All rows from left/right + matches from right/left
# 
# Full Join
# All rows from both with NULL filled values for missing matches columns



# From users table
# Left Join on questions, to find all questions + users with empty ones
# Left Join on answers, to find all answers + users with no answers
# Simply add up these tables
# Group by user, find first time of question and first of answer
three_tables_query = """
                    SELECT u.id AS id,
                        MIN(q.creation_date) AS q_creation_date,
                        MIN(a.creation_date) AS a_creation_date
                    FROM `bigquery-public-data.stackoverflow.users` AS u
                        LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                    ON u.id = a.owner_user_id
                        LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                    ON q.owner_user_id = u.id
                    WHERE u.creation_date >= '2019-01-01' AND u.creation_date < '2019-02-01'
                    GROUP BY id
                    """

# UNION just combines values vertically (joins - horizontally)
# Columns for union should have same type, but column names aren't necessary the same
# UNION ALL - combine two columns including duplicates
# UNION DISTINCT - combine only unique ones, to find total length later, for examlpe.

# Here's find all unique users on specific date who wrote question OR post an answer
# DATE(creation_date) - casting TIMESTAMP to DATE type
all_users_query = """
                  SELECT owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_questions`
                  WHERE DATE(creation_date) = '2019-01-01'
                  UNION DISTINCT
                  SELECT owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_answers`
                  WHERE DATE(creation_date) = '2019-01-01'
                  """

# ANALYTIC FUNCTIONS #
# Three types:

# Aggregate functions:
# MIN (MAX)
# AVG
# SUM
# COUNT

# Navigation functions:
# FIRST_VALUE (LATS_VALUE)
# LEAD (LAG) - return value of subsequent (or preceding) row

# Numbering functions:
# ROW_NUMBER
# RANK


# OVER clause
# PARTITION BY - subset on which make calculations. kind of group by but in-place, 
# ORDER BY
# Window frame - which values to take: 

# ROWS BETWEEN 1 PRECEDING AND CURRENT ROW - the previous row and the current row.
# ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING - the 3 previous rows, the current row, and the following row.
# ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING - all rows in the partition.

# Average amount of number of trips for 15 preceding and next 15 days
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp >= '2016-01-01' AND trip_start_timestamp < '2018-01-01'
                      GROUP BY trip_date
                      ORDER BY trip_date
                      )
                      SELECT trip_date,
                          AVG(num_trips)
                          OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN 15 PRECEDING AND 15 FOLLOWING
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# Create trip_number column with trip number from same area
??? from here until ???END lines may have been inserted/deleted
trip_number_query = """
                    SELECT pickup_community_area,
                        trip_start_timestamp,
                        trip_end_timestamp,
                        RANK()
                            OVER (
                                PARTITION BY pickup_community_area
                                ORDER BY trip_start_timestamp
                            ) AS trip_number
                    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                    WHERE DATE(trip_start_timestamp) = '2017-05-01'
                    """

# Get the break time (between last and next trip)
break_time_query = """
                   SELECT taxi_id,
                       trip_start_timestamp,
                       trip_end_timestamp,
                       TIMESTAMP_DIFF(
                           trip_start_timestamp, 
                           LAG(trip_end_timestamp) 
                               OVER (
                                    PARTITION BY taxi_id 
                                    ORDER BY trip_start_timestamp),
                           MINUTE) as prev_break
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE DATE(trip_start_timestamp) = '2017-05-01' 
                   """



# NESTED AND REPEATED DATA #
# Nested types: STRUCT and RECORD

# Example: `ID`: 1, `Name`: Kek, `Toy`: {`Name`: 'Aby', `Type`: 2}
# To SELECT nested data - just use `.`. e.g 
# SELECT Name as Pet_Name, Toy.Name as Toy_Name, Toy.Type as...

# Select top persons with most commits
max_commits_query = """
                    SELECT committer.name AS committer_name,
                        COUNT(1) AS num_commits
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'
                    GROUP BY committer_name
                    """                    

# REPEATED
# All occurences of one to many relations data can be zipped in REPEATED type (plane array)
# Example: `ID`: 1, `Name`: Kek, `Toys`: [Frisbee, Bone, Rope]

# UNNEST(Toys) - creates new table with 'flatten' values from array, e.g create duplicates of same row with different value


# NESTED + REPEATED
# Example: `ID`: 1, `Name`: Kek, `Toy`: [{`Name`: 'Aby', `Type`: 2}, {`Name`: 'Aby', `Type`: 2}, {`Name`: 'Aby', `Type`: 2}]

# TO `unnest` all:
query = """
        SELECT Name AS Pet_Name,
               t.Name AS Toy_Name,
               t.Tyoe AS Toy_Type
        FROM ..., UNNEST(Toys) AS t
        """

# Get most popular language
# Example row: repo_name: nomemo/LeetCode_C, language: [{'name': 'C', 'bytes': 154785}]
pop_lang_query = """
                    SELECT language.name AS language_name, COUNT(1) AS num_repos
                    FROM `bigquery-public-data.github_repos.languages`, UNNEST(language) as language
                    GROUP BY language.name
                    ORDER BY num_repos DESC
                 """






















































































































































