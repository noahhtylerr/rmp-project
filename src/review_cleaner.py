import re
import html
import ast
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

reviews_df = pd.read_csv('../data/rmp_raw.csv')

before = len(reviews_df)

reviews_df = reviews_df.dropna(subset=['comment'])
reviews_df = reviews_df[reviews_df['comment'].str.strip() != '']
reviews_df = reviews_df[reviews_df['comment'].str.strip() != 'No Comments']

reviews_df = reviews_df.drop_duplicates(subset=['professor_id', 'date', 'comment'])

print('Removed', before - len(reviews_df), 'reviews, kept', len(reviews_df))

reviews_df['comment'] = reviews_df['comment'].apply(html.unescape)

reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_df['year'] = reviews_df['date'].dt.year

# positive review (4 or higher) for classification
reviews_df['positive'] = (reviews_df['quality'] >= 4).astype(int)

reviews_df['textbook'] = reviews_df['textbook'].replace(-1, np.nan)

reviews_df['tags'] = reviews_df['tags'].apply(ast.literal_eval)
reviews_df['tags'] = reviews_df['tags'].apply(lambda tags: [tag.lower() for tag in tags])

print(reviews_df['grade'].value_counts(dropna=False))
print(reviews_df['attendance'].value_counts(dropna=False))
print(reviews_df['department'].value_counts())

stem_depts = ['math', 'physics', 'chemistry', 'biology', 'computer', 'engineering', 'statistics',
              'data science', 'astronomy', 'earth', 'environmental', 'neuroscience', 'optics', 'biomedical',
              'science', 'geology', 'health science', 'nursing', 'medicine', 'information science']
social_depts = ['economics', 'political', 'psychology', 'sociology', 'anthropology', 'business',
                'linguistics', 'public health', 'education', 'brain', 'cognitive', 'social science',
                'accounting', 'marketing', 'finance', 'international', 'geography', 'communication',
                'management']
humanities_depts = ['english', 'history', 'philosophy', 'religion', 'art', 'music', 'language',
                    'literature', 'writing', 'film', 'theater', 'classics', 'french', 'spanish',
                    'german', 'japanese', 'chinese', 'italian', 'russian', 'theology', 'humanities', "women's studies"]

def department_group(department):
    if pd.isna(department):
        return 'Other'

    department = department.lower()
    if any(word in department for word in social_depts):
        return 'Social Science'
    if any(word in department for word in stem_depts):
        return 'STEM'
    if any(word in department for word in humanities_depts):
        return 'Humanities'
    return 'Other'

reviews_df['dept_group'] = reviews_df['department'].apply(department_group)
print(reviews_df['dept_group'].value_counts())

# departments that did not match any group, so you can add them to a list
print(reviews_df.loc[reviews_df['dept_group'] == 'Other', 'department'].value_counts().head(30))

lemmatizer = WordNetLemmatizer()

negation_words = {'not', 'no', 'never', 'nor', 'hardly', 'barely'}

intensifiers = {'super', 'really', 'that', 'too', 'so', 'very', 'extremely', 'overly',
                'particularly', 'especially', 'all', 'at', 'even', 'quite', 'necessarily'}

stop_words = set(stopwords.words('english')) - negation_words

def wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def review_cleaner(text):
    if pd.isna(text):
        return ''

    text = text.lower()
    text = re.sub(r'http\S*', ' ', text)

    # split contractions like isn't, wasn't, can't into a separate 'not'
    text = re.sub(r"n['’]t\b", ' not', text)

    # keep only letters and spaces (removes punctuation, numbers, and emojis)
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = text.split()
    tagged = nltk.pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word, wordnet_tag(tag)) for word, tag in tagged]

    clean_words = []
    negate = False
    for word in lemmas:
        if word in negation_words:
            negate = True
            continue
        if word in stop_words or word in intensifiers or len(word) < 3:
            continue
        if negate:
            word = 'not_' + word
            negate = False
        clean_words.append(word)

    return ' '.join(clean_words)

examples = ["He is not very clear and the exams aren't easy.",
            "Super helpful!! Explains everything clearly, easy A.",
            "Never curves. Tough grader but lectures were engaging."]
for example in examples:
    print(review_cleaner(example))

reviews_df['review_clean'] = reviews_df['comment'].apply(review_cleaner)

print("Empty after cleaning:", (reviews_df['review_clean'] == '').sum())
print(reviews_df[['comment', 'review_clean']].head(10))

grade_points = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0}

# grades like 'Not sure yet', 'Rather not say', 'Drop/Withdrawal' become missing
reviews_df['grade_points'] = reviews_df['grade'].map(grade_points)
reviews_df['attendance'] = reviews_df['attendance'].replace({'Y': 'mandatory', 'N': 'non mandatory'})

# ----- leniency: how generous the grading is -----
leniency_pos_raw = [
    'easy', 'easiest', 'easy a', 'easy grader', 'easy class', 'easy exams', 'easy grade',
    'lenient', 'generous', 'generous grader', 'curve', 'curves', 'curved',
    'extra credit', 'credit opportunities', 'partial credit', 'bonus points', 'easy points',
    'free points', 'drops lowest', 'grade boost', 'chill', 'light workload'
]

leniency_neg_raw = [
    'harsh', 'harshly', 'graded harshly', 'harsh grader', 'harsh grading', 'tough grader',
    'hard grader', 'strict', 'strict grader', 'picky', 'nitpicky', 'unreasonable', 'unfair',
    'subjective', 'inconsistent grading', 'deducts', 'takes off points', 'lose points',
    'low average', 'class average', 'grade deflation', 'impossible to get an a'
]

# ----- teaching quality: how well the professor teaches -----
teaching_pos_raw = [
    'clear', 'clearly', 'explains', 'explains well', 'explains concepts', 'great lecturer',
    'amazing lectures', 'great lectures', 'engaging', 'engaging lectures', 'interesting',
    'organized', 'well organized', 'structured', 'easy to follow', 'easy to understand',
    'passionate', 'knowledgeable', 'helpful', 'helpful feedback', 'great feedback',
    'constructive', 'detailed', 'thorough', 'insightful', 'inspiring', 'inspirational',
    'real life examples', 'approachable', 'answers questions'
]

teaching_neg_raw = [
    'confusing', 'unclear', 'disorganized', 'unorganized', 'boring', 'boring lectures',
    'monotone', 'mumbles', 'rambles', 'rambling', 'tangents', 'reads off slides', 'just reads',
    'reads slides', 'hard to follow', 'hard to understand', 'poor explanations', 'unprepared',
    'pointless', 'useless', 'irrelevant', 'waste of time', 'teach yourself', 'self taught'
]


# clean each word or phrase the same way as the reviews
def clean_word_list(word_list):
    return [review_cleaner(word) for word in word_list if review_cleaner(word) != '']

leniency_pos = clean_word_list(leniency_pos_raw)
leniency_neg = clean_word_list(leniency_neg_raw)
teaching_pos = clean_word_list(teaching_pos_raw)
teaching_neg = clean_word_list(teaching_neg_raw)

# a negated positive word counts as negative, and a negated negative word counts as positive
# (only single words get a negated version, since the cleaner only negates the next word)
leniency_pos = leniency_pos + ['not_' + word for word in leniency_neg if ' ' not in word]
leniency_neg = leniency_neg + ['not_' + word for word in leniency_pos_raw if ' ' not in review_cleaner(word)]
teaching_pos = teaching_pos + ['not_' + word for word in teaching_neg if ' ' not in word]
teaching_neg = teaching_neg + ['not_' + word for word in teaching_pos_raw if ' ' not in review_cleaner(word)]

print(leniency_pos)
print(leniency_neg)
print(teaching_pos)
print(teaching_neg)

reviews_df.to_csv('../data/rmp_clean.csv', index=False)
