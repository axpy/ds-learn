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































































































