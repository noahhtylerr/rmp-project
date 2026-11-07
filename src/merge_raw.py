import ast
import pandas as pd

ratings_df = pd.read_csv('../data/raw/all_ratings.csv')
professors_df = pd.read_csv('../data/raw/all_professors.csv')
print("Reviews:", len(ratings_df), " Professors:", len(professors_df))

# ----- professor columns -----

# drop the empty columns and the school column (the reviews already have it)
professors_df = professors_df.drop(columns=['tags', 'rating_distribution', 'school'])

# rename so professor averages are not confused with each review's own ratings
professors_df = professors_df.rename(columns={
    'id': 'professor_id',
    'name': 'professor_name',
    'overall_rating': 'prof_avg_rating',
    'level_of_difficulty': 'prof_avg_difficulty',
    'percent_take_again': 'prof_take_again_pct',
    'num_ratings': 'prof_num_ratings'
})

# ----- review columns -----

# details is a dictionary saved as text, unpack it into its own columns
ratings_df['details'] = ratings_df['details'].apply(ast.literal_eval)

# create new columns for each detail in review
ratings_df['grade'] = ratings_df['details'].apply(lambda d: d.get('grade'))
ratings_df['attendance'] = ratings_df['details'].apply(lambda d: d.get('attendance'))
ratings_df['for_credit'] = ratings_df['details'].apply(lambda d: d.get('for_credit'))
ratings_df['textbook'] = ratings_df['details'].apply(lambda d: d.get('textbook'))
ratings_df = ratings_df.drop(columns=['details']) # drop details column after unpacking

ratings_df = ratings_df.rename(columns={'course_raw': 'course'})

# merge the professors dataset onto the reviews dataset
raw_df = ratings_df.merge(professors_df, on='professor_id', how='left')

# check nothing was lost or duplicated in the merge
print("Rows after merge:", len(raw_df), " (should match reviews above)")
print("Reviews with no professor match:", raw_df['professor_name'].isna().sum())

# give every review an id
raw_df.insert(0, 'review_id', range(len(raw_df)))

# ----- column order -----
column_order = [
    # ids
    'review_id', 'school', 'professor_id',
    # professor info
    'professor_name', 'department', 'prof_num_ratings',
    'prof_avg_rating', 'prof_avg_difficulty', 'prof_take_again_pct',
    # this review
    'course', 'date', 'quality', 'difficulty',
    'grade', 'attendance', 'for_credit', 'textbook',
    'tags', 'thumbs_up', 'thumbs_down',
    # review text last since it is long
    'comment'
]
raw_df = raw_df[column_order]

raw_df.to_csv('../data/interim/rmp_raw.csv', index=False)
print("Saved", raw_df.shape)
raw_df.head()