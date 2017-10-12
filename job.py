from get_arxiv import update_arxiv
from tweet_papers import tweet_latest_day

update_arxiv()
tweet_latest_day(dry_run=False)
