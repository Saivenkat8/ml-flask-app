import json
import requests
from requests_oauthlib import OAuth1
from sys import argv

def get_tweets(*args, count=10):
    url = "https://api.twitter.com/1.1/search/tweets.json"
    tweets = []

    for search_terms in args:
        params = {"q": search_terms, "lang": "en", "count": count, "tweet_mode": "extended"}

        response = requests.get(url, auth=auth, params=params)
        tweets += response.json()["statuses"]

    return tweets

with open("twitter_credentials.json", "r") as f:
    secrets = json.load(f)

auth = OAuth1(secrets["CONSUMER_KEY"],
              secrets["CONSUMER_SECRET"],
              secrets["ACCESS_TOKEN"],
              secrets["ACCESS_SECRET"])

tweets = get_tweets("python")

for tweet in tweets:
    r = requests.get("http://127.0.0.1:5000/predict", params={"tweet": tweet['full_text']})
    print(tweet['full_text'])
    print(r.text)
