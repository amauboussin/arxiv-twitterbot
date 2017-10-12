import os

TWEET_THRESHOLD = .5
MAX_TWEET_LENGTH = 140  # holding out for 280
TIME_BETWEEN_TWEETS = 2

BASE_URL = 'http://export.arxiv.org/api/query'
KARPATHY_CATEGORIES = ['cs.CV', 'cs.AI', 'cs.LG', 'cs.CL', 'cs.NE', 'stat.ML']
BRUNDAGE_CATEGORIES = ['cs*', 'cond-mat.dis-nn', 'q-bio.NC', 'stat.CO', 'stat.ML']

current_directory = os.path.dirname(os.path.abspath(__file__))

PICKLE_PATH = os.path.join(current_directory, 'all_arxiv.pickle')
TWEET_CSV_PATH = os.path.join(current_directory, 'miles_links.csv')
DICT_TRANSFORM_PATH = os.path.join(current_directory, 'dictionary.json')
MODEL_PATH = os.path.join(current_directory, 'conv_weights.h5')

QUERY_PAGE_SIZE = 500
QUERY_WAIT_TIME = 10  # in seconds
