import praw
import json

reddit = praw.Reddit(
    client_id='cTDyei_VFQZdEk3nkAM9Kg',
    client_secret='{SECRET}',
    user_agent='Scraping:App for scraping for learning purposes:v1.0 (by u/wikator123)'
)

subreddit_name = 'poland'
num_posts = 100

# Pobieranie postów
subreddit = reddit.subreddit(subreddit_name)
posts = []

for submission in subreddit.new(limit=num_posts):
    post = {
        'title': submission.title,
        'score': submission.score,
        'id': submission.id,
        'url': submission.url,
        'comms_num': submission.num_comments,
        'created': submission.created,
        'body': submission.selftext,
    }
    posts.append(post)

with open('reddit_posts.json', 'w', encoding='utf-8') as f:
    json.dump(posts, f, ensure_ascii=False, indent=4)

print(f"Pobrano {num_posts} postów z subreddita r/{subreddit_name} i zapisano do pliku reddit_posts.json")
