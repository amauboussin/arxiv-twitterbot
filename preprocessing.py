
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from constants import PICKLE_PATH, TWEET_CSV_PATH

# python -m spacy download en


def load_arxiv_and_tweets():
    arxiv = pd.read_pickle(PICKLE_PATH)

    arxiv['link'] = arxiv.link.apply(clean_arxiv_api_link)

    miles_links = pd.read_csv(TWEET_CSV_PATH)
    miles_links['time'] = miles_links.time.apply(pd.Timestamp)
    miles_links['link'] = miles_links['link'].apply(clean_miles_link)
    df = miles_links.set_index('link').join(arxiv.set_index('link'), how='right')

    df = df.reset_index().groupby('link').apply(group_tweeted_multiple).reset_index(drop=True)
    df = df.assign(tweeted=(~df.time.isnull()).astype(int))

    # remove papers past the day of the last tweet
    return df


def get_sklearn_data():
    """Get data for training an sklearn model"""
    df = load_arxiv_and_tweets().sort_values('published')
    max_date = df[df.tweeted == 1].published.max()
    return get_features_matrix(df[df.published < max_date])


def get_tokenized_list_of_dicts():
    """Get data as a list of dictionaries with spacy docs + labels for training the conv net"""
    df = load_arxiv_and_tweets()
    max_date = df[df.tweeted == 1].published.max()
    data_dicts = arxiv_df_to_list_of_dicts(df[df.published < max_date])
    tokenized_data = parse_content_serial(data_dicts)
    return tokenized_data


def get_features_matrix(df, min_author_freq=3, min_term_freq=30, ngram_range=(1, 3)):
    """Return numpy array of data for sklearn models"""
    text = [title + ' ' + summary for title, summary in zip(df.title.values, df.summary.values)]
    vectorizer = TfidfVectorizer(min_df=min_term_freq, stop_words='english', ngram_range=ngram_range)
    text_features = vectorizer.fit_transform(text).toarray()

    author_counts = pd.Series([a for author_set in df.authors.values for a in author_set]).value_counts()
    allowed_authors = author_counts[author_counts >= min_author_freq].index
    filtered_authors = df.authors.apply(lambda authors: [a for a in authors if a in allowed_authors])

    author_binarizer = MultiLabelBinarizer()
    author_features = author_binarizer.fit_transform(filtered_authors.values)

    category_dummies = pd.get_dummies(df.category)
    category_features = category_dummies.values

    all_features = [text_features, author_features, category_features]

    x = np.concatenate(all_features, axis=1)

    if 'tweeted' in df:
        y = df.tweeted.astype(int).values
    else:
        y = None

    feature_names = np.concatenate((vectorizer.get_feature_names(),
                                    category_dummies.columns.values,
                                    author_binarizer.classes_))
    return x, y, feature_names


def get_spacy_parser():
    return spacy.load('en')


def group_tweeted_multiple(df):
    row = df.iloc[0]
    if df.shape[0] > 1:
        row[['rts', 'favorites']] = df.rts.sum(), df.favorites.sum()
    return row


def arxiv_df_to_list_of_dicts(df):

    def row_to_example(row):
        def to_token(s):
            """Squash a string into one token by removing non-alpha characters"""
            return ''.join([c for c in s if c.isalpha()])

        category_token = to_token(row.category)
        author_tokens = ' '.join([to_token(author) for author in row.authors])
        to_concat = [row.title, row.summary, author_tokens, category_token]
        text = ' '.join(to_concat).replace('\n', ' ')
        return {
            'label': row.tweeted,
            'id': row['index'],
            'content': text,
            'link': row.link
        }
    return [row_to_example(row) for i, row in df.reset_index().iterrows()]


def clean_arxiv_api_link(link):
    if not link[-1].isdigit():
        return None
    return link.replace('http://', '').replace('https://', '')[:-2]


def clean_miles_link(link):
    if not link[-1].isdigit():
        return None
    return link.replace('http://', '').replace('https://', '')


def parse_content_serial(data):
    """Parse the content field of a list of dicts from unicode to a spacy doc"""
    spacy_parser = get_spacy_parser()
    for row in data:
        row['content'] = spacy_parser(row['content'])
    return data


def sorted_train_test_split(x, y, test_size):
    train_size = 1. - test_size
    train_end_index = int(len(x) * train_size)
    return x[:train_end_index], x[train_end_index:], y[:train_end_index], y[train_end_index:]

