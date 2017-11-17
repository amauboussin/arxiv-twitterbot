from time import sleep

from get_arxiv import check_for_update, update_arxiv
from tweet_papers import tweet_day

if check_for_update():
    update_arxiv()
    sleep(2)
    tweet_day(dry_run=False)
