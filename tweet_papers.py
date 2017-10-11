from time import sleep

import pandas as pd
import tweepy

from constants import MAX_TWEET_LENGTH, TWEET_THRESHOLD, TIME_BETWEEN_TWEETS
from credentials import *
from predict import add_predictions_to_date
from preprocessing import load_arxiv_and_tweets


def get_title_tweet(date):
    return 'arXiv papers published on {}:'.format(date.strftime('%B %-m, %Y'))


def tweet_latest_day(dry_run=True):
    df = load_arxiv_and_tweets()
    index_to_predict_for = get_published_on_day_index(df)
    df = add_predictions_to_date(df, index_to_predict_for)
    predicted_papers = df.loc[index_to_predict_for]
    to_tweet = predicted_papers[predicted_papers.prediction > TWEET_THRESHOLD]
    if not to_tweet.empty:
        paper_tweets = to_tweet.sort_values('prediction', ascending=False).apply(get_tweet_text, axis=1)
        title_tweet = get_title_tweet(to_tweet.iloc[0].published.date())
        to_tweet = [title_tweet] + list(paper_tweets.values)

        if dry_run:
            for t in to_tweet:
                print t
                print
        else:
            in_reply_to = None
            api = get_tweepy_api()
            for t in to_tweet:
                last_tweet = api.update_status(t, in_reply_to)
                in_reply_to = last_tweet.id
                sleep(TIME_BETWEEN_TWEETS)


def get_published_on_day_index(df, date=None):
    """Return an index that locates paper published on the given date"""
    if date is None:
        date = df.published.dt.date.max()
    else:
        date = pd.Timestamp(date).date()
    return df[df.published.dt.date == date].index


def get_tweepy_api():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def truncate_at_whitespace(text, max_len):
    tokens = text.split()
    truncated = []
    cur_len = 0
    for i, token in enumerate(tokens):
        cur_len += len(tokens[i]) + 1
        if cur_len > max_len:
            break
        truncated.append(token)
    return ' '.join(truncated)


def get_tweet_text(paper):
    """Given a series with info on a paper, compose a tweet"""
    title = paper.title.replace('\n ', '')
    link = paper.link
    last_names = [a.split(' ')[-1] for a in paper.authors]

    if len(paper.authors) > 1:
        oxford_comma = u',' if len(paper.authors) > 2 else u''
        authors = u', '.join(last_names[:-1]) + oxford_comma + u' and ' + last_names[-1]
    else:
        authors = paper.authors[0]

    full_tweet = u'{title}, {authors} {link}'.format(title=title, authors=authors, link=link)
    if len(full_tweet) < 140:
        return full_tweet

    authors_et_al = last_names[0] + u' et al.'
    short_author_tweet = u'{title}, {authors} {link}'.format(title=title, authors=authors_et_al,
                                                                link=link)
    if len(short_author_tweet) < 140:
        return short_author_tweet

    sans_title = u'{authors} {link}'.format(authors=authors_et_al, link=link)
    max_title_len = MAX_TWEET_LENGTH - 5 - len(sans_title)
    truncated_title = truncate_at_whitespace(title, max_title_len)
    return truncated_title + u'... ' + sans_title
