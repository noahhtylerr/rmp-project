import re
import pandas as pd

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ----- department groups -----

social_depts = ['economics', 'political', 'psychology', 'sociology', 'anthropology', 'business',
                'linguistics', 'public health', 'education', 'brain', 'cognitive', 'social science',
                'accounting', 'marketing', 'finance', 'international', 'geography', 'communication',
                'management', 'criminal justice', 'social work', 'physical ed', 'american studies']
stem_depts = ['math', 'physics', 'chemistry', 'biology', 'computer', 'engineering', 'statistics',
              'data science', 'astronomy', 'earth', 'environmental', 'neuroscience', 'optics', 'biomedical',
              'science', 'geology', 'health science', 'nursing', 'medicine', 'information science',
              'information technology', 'computing']
humanities_depts = ['english', 'history', 'philosophy', 'religion', 'art', 'music', 'language',
                    'literature', 'writing', 'film', 'theater', 'classics', 'french', 'spanish',
                    'german', 'japanese', 'chinese', 'italian', 'russian', 'theology', 'humanities',
                    "women's studies", 'design', 'journalism', 'hospitality']


# assigns a department to Social Science, STEM, Humanities, or Other by keyword
def department_group(department):
    if pd.isna(department):
        return 'Other'

    department = department.lower()
    # social science is checked first so names like "Political Science" are not caught by "science" in STEM
    if any(word in department for word in social_depts):
        return 'Social Science'
    if any(word in department for word in stem_depts):
        return 'STEM'
    if any(word in department for word in humanities_depts):
        return 'Humanities'
    return 'Other'


# letter grades as numbers -> grades like 'Not sure yet', 'Rather not say', 'Drop/Withdrawal' become missing
grade_points = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0}

# ----- review cleaner -----

# ===== review cleaner =====
# reduces each word to its dictionary form
lemmatizer = WordNetLemmatizer()

# words that flip the meaning of the next real word
negation_words = {'not', 'no', 'never', 'nor', 'hardly', 'barely'}

# filler words between 'not' and the real word, negation skips over these
intensifiers = {'super', 'really', 'that', 'too', 'so', 'very', 'extremely', 'overly',
                'particularly', 'especially', 'all', 'at', 'even', 'quite', 'necessarily'}

# english stopwords, minus the negation words we need to keep
stop_words = set(stopwords.words('english')) - negation_words


# converts NLTK part of speech tags into the format WordNet uses
def wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# cleans one review, lowercases, removes punctuation, lemmatizes, and marks negation (not clear -> not_clear)
def review_cleaner(text):
    # missing text returns an empty string
    if pd.isna(text):
        return ''

    # lowercase the text and remove links
    text = text.lower()
    text = re.sub(r'http\S*', ' ', text)

    # split contractions like isn't, wasn't, can't into a separate 'not'
    text = re.sub(r"n['’]t\b", ' not', text)

    # keep only letters and spaces (removes punctuation, numbers, and emojis)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # tag each word's part of speech, then lemmatize it
    words = text.split()
    tagged = nltk.pos_tag(words)
    lemmas = [lemmatizer.lemmatize(word, wordnet_tag(tag)) for word, tag in tagged]

    # attach negation to the next real word, and drop stopwords, filler words, and short words
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


# ----- word lists -----

# leniency: how generous the grading is
leniency_pos_raw = [
    'easy', 'easiest', 'easy a', 'easy grader', 'easy class', 'easy exams', 'easy grade',
    'lenient', 'generous', 'generous grader', 'curve', 'curves', 'curved',
    'extra credit', 'credit opportunities', 'partial credit', 'bonus points', 'easy points',
    'free points', 'drops lowest', 'grade boost', 'chill', 'light workload'
]

leniency_neg_raw = [
    'harsh', 'harshly', 'graded harshly', 'harsh grader', 'harsh grading', 'tough grader',
    'hard grader', 'strict', 'strict grader', 'picky', 'unreasonable', 'unfair',
    'subjective', 'inconsistent grading', 'deducts', 'takes off points', 'lose points',
    'low average', 'class average', 'grade deflation', 'impossible to get an a'
]

# teaching quality: how well the professor teaches
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
    'monotone', 'mumbles', 'rambles', 'rambling', 'tangents', 'reads off slides',
    'reads slides', 'hard to follow', 'hard to understand', 'poor explanations', 'unprepared',
    'pointless', 'useless', 'irrelevant', 'waste of time', 'self taught'
]


# clean each word or phrase the same way as the reviews, skipping duplicates
def clean_word_list(word_list):
    cleaned = []
    for word in word_list:
        clean = review_cleaner(word)
        if clean != '' and clean not in cleaned:
            cleaned.append(clean)
    return cleaned


# builds the final positive and negative lists for one concept
def build_word_lists(pos_raw, neg_raw):
    pos = clean_word_list(pos_raw)
    neg = clean_word_list(neg_raw)
    # a negated positive word counts as negative (not_clear), a negated negative word counts as positive (not_harsh)
    pos_final = pos + ['not_' + word for word in neg if ' ' not in word]
    neg_final = neg + ['not_' + word for word in pos if ' ' not in word]
    return pos_final, neg_final


leniency_pos, leniency_neg = build_word_lists(leniency_pos_raw, leniency_neg_raw)
teaching_pos, teaching_neg = build_word_lists(teaching_pos_raw, teaching_neg_raw)
