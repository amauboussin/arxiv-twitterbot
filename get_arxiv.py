import os
from time import sleep

import feedparser
import pandas as pd
import requests

from constants import BASE_URL, BRUNDAGE_CATEGORIES, PICKLE_PATH, QUERY_PAGE_SIZE, QUERY_WAIT_TIME




def get_entry_dict(entry):
    """Return a dictionary with the items we want from a feedparser entry"""
    try:
        return {
            'title': entry['title'],
            'authors': [a['name'] for a in entry['authors']],
            'published': pd.Timestamp(entry['published']),
            'summary': entry['summary'],
            'link': entry['link'],
            'category': entry['category'],
        }
    except KeyError:
        print('Missing keys in row: {}'.format(entry))
        return None


def strip_version(link):
    return link[:-2]


def load():
    """Load data from pickle and remove duplicates"""
    df = pd.read_pickle(PICKLE_PATH)
    return (df.sort_values('published', ascneding=False)
            .groupby('unique_link').first().reset_index())


def fetch_updated_data(prev_data):
    """Get new paper from arxiv server"""

    def make_query_string(categories):
        return '+OR+'.join(['cat:' + c for c in categories])

    past_links = prev_data.link.apply(strip_version)
    i = 0
    while True:
        params = {
            'search_query': make_query_string(BRUNDAGE_CATEGORIES),
            'sortBy': 'submittedDate',
            'start': QUERY_PAGE_SIZE * i,
            'max_results': QUERY_PAGE_SIZE,
        }
        param_string = 'search_query={search_query}&sortBy={sortBy}&start={start}&max_results={max_results}'.format(
            **params)
        response = requests.get(BASE_URL, params=param_string)
        parsed = feedparser.parse(response.text)
        entries = parsed.entries

        if len(entries) == 0:
            continue

        parsed_entries = [get_entry_dict(e) for e in entries]

        results_df = pd.DataFrame(parsed_entries)
        print ('Fetched {} abstracts published {} and earlier').format(results_df.shape[0],
                                                                       results_df.published.max().date())

        new_links = ~results_df.link.apply(strip_version).isin(past_links)
        import pdb; pdb.set_trace()
        if not new_links.any():
            break

        prev_data = pd.concat((prev_data, results_df.loc[new_links]))

        i += 1
        sleep(QUERY_WAIT_TIME)

    return prev_data


def update_arxiv():
    """Update arxiv data pickle witht the latest abstracts"""
    prev_data = pd.read_pickle(PICKLE_PATH)
    updated_data = fetch_updated_data(prev_data)
    print ('Downloaded {} new abstracts'.format(updated_data.shape[0] - prev_data.shape[0]))
    updated_data.to_pickle('all_arxiv.pickle')