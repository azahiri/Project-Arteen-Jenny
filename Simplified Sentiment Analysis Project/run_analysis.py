# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import pickle           # For pickling data

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
'exec(%matplotlib inline)'
import time
import datetime

# For sentiment analysis:
from textblob import TextBlob
import re

# We import our access keys:
from credentials import *    # This allows us to use the keys as variables

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

def get_tweets_by_username(screen_name):
    """We create an extractor object:"""
    extractor = twitter_setup()
    tweets = tweepy.Cursor(extractor.user_timeline, screen_name=screen_name)

    return tweets

def process_tweets(tweets, keywords):
    """
    Given a list of tweets, a list of keywords,
    return a list of tweets that only contain the keywords
    """
    useful_tweets = []
    for tweet in tweets:
        for keyword in keywords:
            if keyword in tweet.text:
                useful_tweets.append(tweet)
                break
    return useful_tweets

def show_text(tweets):
    """
    Given a list of tweets
    return a list of tweet texts
    """
    arsenal_tweet_text = []
    for tweet in tweets:
        arsenal_tweet_text.append(tweet.text)
    return arsenal_tweet_text

def show_len(tweets):
    """
    Given a list of tweets
    return a list of tweet lengths
    """
    arsenal_tweet_len = []
    for tweet in tweets:
        arsenal_tweet_len.append(len(tweet.text))
    return arsenal_tweet_len

def show_date(tweets):
    """
    Given a list of tweets
    return a list of tweet dates
    """
    arsenal_tweet_date = []
    for tweet in tweets:
        arsenal_tweet_date.append(tweet.created_at.strftime('%m/%d/%Y'))
    return arsenal_tweet_date

def show_ID(tweets):
    """
    Given a list of tweets
    return a list of tweet IDs
    """
    arsenal_tweet_text = []
    for tweet in tweets:
        arsenal_tweet_text.append(tweet.id)
    return arsenal_tweet_text

def show_source(tweets):
    """
    Given a list of tweets
    return a list of tweet sources
    """
    arsenal_tweet_text = []
    for tweet in tweets:
        arsenal_tweet_text.append(tweet.source)
    return arsenal_tweet_text

def show_likes(tweets):
    """
    Given a list of tweets
    return a list of tweet likes
    """
    arsenal_tweet_text = []
    for tweet in tweets:
        arsenal_tweet_text.append(tweet.favorite_count)
    return arsenal_tweet_text

def show_RT(tweets):
    """
    Given a list of tweets
    return a list of tweet retweets
    """
    arsenal_tweet_text = []
    for tweet in tweets:
        arsenal_tweet_text.append(tweet.retweet_count)
    return arsenal_tweet_text

def obtain_sources(data_table, SCREEN_NAME):
    """
    Given a panda dataframe
    print all tweet creation sources
    """
    sources = []
    for source in data_table['Source']:
        if source not in sources:
            sources.append(source)
    print(f"@{SCREEN_NAME}'s tweet creation sources:")
    for source in sources:
        print("* {}".format(source))


def clean_tweet(tweet):
    """
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analyze_polarity(tweet):
    """
    Utility function to classify the polarity of a tweet
    using textblob.
    '"""
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

def analyze_subjectivity(tweet):
    """
    Utility function to classify the subjectivity of a tweet
    using textblob.
    '"""
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.subjectivity > 0:
        return 1
    else:
        return 0

def main():
    input_keywords = input("Enter a list keywords separated by comma: ")
    KEYWORDS = input_keywords.split(',')
    SCREEN_NAME = input('Screen Name: @')
    
    tweets = get_tweets_by_username(screen_name=SCREEN_NAME)

    tweet_list = list(tweets.items())

    with open(f'tweets_by_{SCREEN_NAME}.p', 'wb') as f:
        pickle.dump(tweet_list, f)

    with open(f'tweets_by_{SCREEN_NAME}.p', 'rb') as f:
        tweet_list = pickle.load(f)

    useful_tweets = process_tweets(tweet_list, KEYWORDS)
    useful_tweets_text = show_text(useful_tweets)
    useful_tweets_len = show_len(useful_tweets)
    useful_tweets_date = show_date(useful_tweets)
    useful_tweets_ID = show_ID(useful_tweets)
    useful_tweets_source = show_source(useful_tweets)
    useful_tweets_likes = show_likes(useful_tweets)
    useful_tweets_RT = show_RT(useful_tweets)

    print("Total number of tweets about these keywords: {} tweets\n".format(len(useful_tweets)))

    # We create a pandas dataframe as follows:
    data = pd.DataFrame(data=useful_tweets_text, columns=['Tweets'])

    # We add relevant data:
    data['len']  = np.array(useful_tweets_len)
    data['ID']   = np.array(useful_tweets_ID)
    data['Date'] = np.array(useful_tweets_date)
    data['Source'] = np.array(useful_tweets_source)
    data['Likes']  = np.array(useful_tweets_likes)
    data['RTs']    = np.array(useful_tweets_RT)

    # We extract the mean of lenghts:
    mean = np.mean(data['len'])

    print("Average length of tweet: {0:.2f} characters\n".format(float(mean)))

    # We extract the tweet with more FAVs and more RTs:
    fav_max = np.max(data['Likes'])
    rt_max  = np.max(data['RTs'])

    fav = data[data.Likes == fav_max].index[0]
    rt  = data[data.RTs == rt_max].index[0]

    # Max FAVs:
    print("The tweet with the most likes is: \n{}".format(data['Tweets'][fav]))
    print("Number of likes: {}".format(fav_max))
    print("{} characters.\n".format(data['len'][fav]))

    # Max RTs:
    print("The tweet with the most retweets is: \n{}".format(data['Tweets'][rt]))
    print("Number of retweets: {}".format(rt_max))
    print("{} characters.\n".format(data['len'][rt]))

    obtain_sources(data, SCREEN_NAME)
    
    # We create a column with the result of the analysis:
    data['Pol'] = np.array([ analyze_polarity(tweet) for tweet in data['Tweets'] ])
    data['Sub'] = np.array([ analyze_subjectivity(tweet) for tweet in data['Tweets'] ])

    # We display the updated dataframe with the new column:
    print("\nLast 10 Tweets:")
    display(data.head(10))

    # We construct lists with classified tweets:
    pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['Pol'][index] > 0]
    neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['Pol'][index] == 0]
    neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['Pol'][index] < 0]

    # We print percentages:
    print("\nPercentage of positive tweets: {0:.2f}%".format(float(len(pos_tweets)*100/len(data['Tweets']))))
    print("Percentage of neutral tweets: {0:.2f}%".format(float(len(neu_tweets)*100/len(data['Tweets']))))
    print("Percentage of negative tweets: {0:.2f}%\n".format(float(len(neg_tweets)*100/len(data['Tweets']))))

    # We create time series for data:
    tlen = pd.Series(data=data['len'].values, index=data['Date'])
    tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
    tret = pd.Series(data=data['RTs'].values, index=data['Date'])

    # Tweet Lenght Visualization:
    tlen.plot(figsize=(16,4), color='r')
    print("Length of tweets over time:")
    plt.show()

    # Likes vs Retweets Visualization:
    tfav.plot(figsize=(16,4), label="Likes", legend=True)
    tret.plot(figsize=(16,4), label="Retweets", legend=True)
    print("Likes vs Retweets on tweets over time:")
    plt.show()

    print("Hope you enjoyed those visuals!")

    print("\nFor your reference...")
    print("Polarity is a float within the range [-1.0, 1.0], -1.0 being completely negative and 1.0 being completely positive.")
    print("Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.")

if __name__ == '__main__':
    main()