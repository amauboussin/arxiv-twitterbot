from time import sleep

import pandas as pd
import tweepy

from constants import MAX_TWEET_LENGTH, PAST_PREDICTIONS_PATH, TWEET_THRESHOLD, TIME_BETWEEN_TWEETS
from credentials import access_token, access_token_secret, consumer_key, consumer_secret
from predict import add_conv_predictions_to_date
from preprocessing import load_arxiv_and_tweets


def tweet_latest_day(dry_run=True, check_if_most_recent=True):
    """Get predictions and tweet papers for the papers published on max_publiscation_date"""
    df = load_arxiv_and_tweets()
    past_predictions = pd.read_pickle(PAST_PREDICTIONS_PATH)

    index_to_predict_for = get_latest_without_prediction(df, past_predictions)
    if index_to_predict_for.size == 0:
        raise Exception('No new papers found')
    df = add_conv_predictions_to_date(df, index_to_predict_for)
    if not dry_run:
        preds = df.loc[~df.prediction.isnull()][['link', 'prediction']]
        pd.concat((past_predictions, preds)).to_pickle(PAST_PREDICTIONS_PATH)

    predicted_papers = df.loc[index_to_predict_for]
    to_tweet = predicted_papers[predicted_papers.prediction > TWEET_THRESHOLD]
    if not to_tweet.empty:
        published_on = to_tweet.iloc[0].published.date()
        paper_tweets = to_tweet.sort_values('prediction', ascending=False).apply(get_tweet_text, axis=1)
        title_tweet = get_title_tweet(one_weekday_later(pd.Timestamp(published_on)))
        to_tweet = [title_tweet] + list(paper_tweets.values) + ['Fin.']

        if dry_run:
            for t in to_tweet:
                print t
                print

        elif check_if_most_recent and published_on < most_recent_weekday():
            print "Don't have any new papers for today, latest are from {}".format(published_on)
            return

        else:
            in_reply_to = None
            api = get_tweepy_api()
            print 'Tweeting {} papers published on {}'.format(len(to_tweet), published_on)
            for t in to_tweet:
                last_tweet = api.update_status(t, in_reply_to)
                in_reply_to = last_tweet.id
                sleep(TIME_BETWEEN_TWEETS)
            print 'Done'


def get_title_tweet(published_date=None):
    if published_date is None:
        published_date = pd.Timestamp('now') + pd.Timedelta(days=1).date()
    return 'arXiv papers, {}:'.format(published_date.strftime('%B %-d, %Y'))


def get_latest_without_prediction(df, predictions):
    joined = predictions.set_index('link').join(df.set_index('link'), how='inner')
    max_pub_time = joined.published.max()
    print 'Tweeting papers from {} on'.format(max_pub_time)
    return df[df.published > max_pub_time].index


def get_published_on_day_index(df, date=None):
    """Return the index for papers published on the given date"""
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

    full_tweet = u'{title}. {authors} {link}'.format(title=title, authors=authors, link=link)
    if len(full_tweet) < 140:
        return full_tweet

    authors_et_al = last_names[0] + u' et al.'
    short_author_tweet = u'{title}. {authors} {link}'.format(title=title, authors=authors_et_al,
                                                             link=link)
    if len(short_author_tweet) < 140:
        return short_author_tweet

    sans_title = u'{authors} {link}'.format(authors=authors_et_al, link=link)
    max_title_len = MAX_TWEET_LENGTH - 4 - len(sans_title)
    truncated_title = truncate_at_whitespace(title, max_title_len)
    return truncated_title + u'... ' + sans_title


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


def most_recent_weekday():
    """Return today (if today is on a weekday) or the friday beforehand if it is a weekend"""
    dt = pd.Timestamp('now')
    while dt.weekday() > 4:  # Mon-Fri are 0-4
        dt = dt - pd.Timedelta(days=1)
    return dt.date()


def one_weekday_later(dt):
    dt = dt + pd.Timedelta(days=1)
    while dt.weekday() > 4:  # Mon-Fri are 0-4
        dt = dt + pd.Timedelta(days=1)
    return dt.date()
