#import necessary libraries 
import tweepy 
from textblob import TextBlob
import datetime
import re
import numpy as np 
import matplotlib.pyplot as plt
import os
##Functions to Validate and Clean 

image_folder = os.path.join('static', 'images')

# function to remove url! 
def remove_url(txt):
    """
    function that removes url, which will be used later for the 
    tweet cleaning purpose
    """
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


#authenticate the keys 
def auth():
    consumer_key = 'WWSMyFjVOJzjR5wtXizC1B0go'
    consumer_secret = 'w1DsVNTFc4v9fdtSItr3V8O2JVsleSHlvhdkhckoSit6AKP9XP'
    access_token = '1115278543863386112-aqOKS4yOFao6ZUe6PaQwWxYOJyk3gP' 
    acccess_token_secret = 'HKHbROYbNNxP50GsluZ1wf4h6GjAhSKh6j69GOWGFnilW' 

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, acccess_token_secret)
    api = tweepy.API(auth)
    return api


#Create second search
def main():
    """
    second_search() function gives users the option to search another keyword to be analyzed. 
    if users respond 'yes' to the first question, it will allow the users to input another keyword. 
    The output would provide sensitivity analysis for both keywords.   
    """
    while True:
        user_response = input('Would you like to analyze another term? Please input yes or no : ')

        if str(user_response) in ['yes', 'no']:
            if user_response == 'no':
                print_items(user_input, cleaned_tweets, analysis)
                break
            else:
                user_input2= input('What topic would you like to analyze as your second keyword? ')
       
                public_tweets2 = api.search(user_input2, count = 1000)
                dictionary_tweets2 = create_dictionary(public_tweets2)
                cleaned_tweets2 = clean_tweets(dictionary_tweets2)
                analysis2 = get_sentiment(cleaned_tweets2)
                polarity2 = get_sentiment(cleaned_tweets2)[0]
                subjectivity2 = get_sentiment(cleaned_tweets2)[1]
                print_items(user_input, cleaned_tweets, analysis)
                print_items(user_input2, cleaned_tweets2, analysis2)
                bar_plot(polarity1, subjectivity1, polarity2, subjectivity2, user_input, user_input2)
                break


# # Step 3 - Retrieve Tweets
# user_input = input('What topic would you like to analyze as your keyword? ')
# public_tweets = api.search(user_input, count = 1000)

# search = input(str("What search term do you want to search? "))
# public_tweets=api.search(search)


# print(public_tweets)
def create_dictionary(public_tweets):
    """
    Create a dictionary that includes only the necessary items from the twitter database.
    We only need the unique ID and the actual tweets.
    
    """
    tweet_dictionary={}
    for tweet in public_tweets:
        # print(dir(tweet))
        # print(tweet.text, tweet.id)
        # print(dir(tweet.user))
        # print(tweet.user.screen_name)
        key = tweet.user.screen_name
        if key not in tweet_dictionary:
            tweet_dictionary[key] = [tweet.text]
        else: 
            tweet_dictionary[key].append(tweet.text)
    return tweet_dictionary

# print(create_dictionary(public_tweets))

# dictionary_tweets = create_dictionary(public_tweets)




# tweets = api.user_timeline('realDonaldTrump')
# print(dir(tweets[0]))
# print(tweets[0].text)
# print(tweets[0].id, tweets[0].text, tweets[0].created_at)

# to print tweets
# CLEANS LIST OF DICTIONARY
def clean_tweets(dictionary_tweets):
    """
    function that cleans the tweets 
    this function removes the user id, url, and the 'RT' sign from 
    the values of the dictionary tweets 
    """
    for key, value in dictionary_tweets.items():
        count = 0
        for word in value:
            # print(words)
            # print(type(words))
            word = re.sub('@[^\s]+','',word) #gets rid of id 
            word = remove_url(word) #gets rid of url
            word = word.strip('RT') #get rid of rt
            dictionary_tweets[key][count] = word
            count += 1
    # print(dictionary_tweets)
    return dictionary_tweets

# cleaned_tweets = clean_tweets(dictionary_tweets)

# print(cleaned_tweets)
# print({key:value for key, value in cleaned_tweets.items() if len(value) > 1}) 
#^ to check if there's more than one value







# Get sentiment analysis 
def get_sentiment(cleaned_tweets):
    """
    this function provides the polarity and the sensitivity 
    of all the tweets collected in the dictionary 
    """
    total_polarity = 0
    total_subjectivity = 0
    num_tweets = 0
    for value in cleaned_tweets.values(): 
        for tweet in value:
            text = TextBlob(tweet)
            total_polarity += text.sentiment.polarity
            total_subjectivity += text.sentiment.subjectivity
            num_tweets += 1
    avg_polarity = total_polarity / num_tweets
    avg_subjectivity = total_subjectivity / num_tweets
    return avg_polarity, avg_subjectivity

# polarity1 = get_sentiment(cleaned_tweets)[0]
# subjectivity1= get_sentiment(cleaned_tweets)[1]

# print(f"Polarity: {get_sentiment(cleaned_tweets)[0]:.3f}", f"Subjectivity: {get_sentiment(cleaned_tweets)[1]:.3f}")
# print(f"Polarity: {get_sentiment(cleaned_tweets2)[0]:.3f}", f"Subjectivity: {get_sentiment(cleaned_tweets2)[1]:.3f}")
# sentiment = get_sentiment(dictionary_tweets)
# where polarity is a float within the range [-1.0, 1.0] 
# and subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.



# #write a function that tells us if the tweet is positive or not 
def get_label(analysis, threshold=0): # threshold
    """
    this function provides the sentiment of the tweets.
    It tells us if the overall tweets 
    were positive of negative 
    """
    if analysis[0] > threshold:
       return 'Positive'
    else:
        return 'Negative'

# analysis = get_sentiment(cleaned_tweets)

# print(get_label(analysis, threshold=0))


def bar_plot(polarity1, subjectivity1, polarity2, subjectivity2, user_input1, user_input2):
    """
    Bar_plot that provides the visualized comparison of the polarity and subjectivity between 
    the two user inputs. Great tools for users to see the different sentiments between two inputs. 
    """
    n_group = 2
    comparison1 = (polarity1, subjectivity1)
    comparison2 = (polarity2, subjectivity2)

    #create plot 
    fig, ax= plt.subplots()
    index = np.arange(n_group)
    bar_width = 0.35 
    opacity = 0.8 

    rects1 = plt.bar(index, comparison1, bar_width,
    alpha = opacity,
    color = 'b',
    label = user_input1)
    
    rects2 = plt.bar(index + bar_width, comparison2, bar_width,
    alpha = opacity,
    color = 'r',
    label = user_input2)

    plt.ylabel('Score')
    plt.title('Polarity and Subjectivity Comparison')
    plt.xticks(index + bar_width/2, ('Polarity', 'Subjectivity'))
    plt.legend()

    plt.tight_layout
    plt.savefig(os.path.join(image_folder, 'barplot.png'))

# bar_plot(0.50, 0.70, 0.45, 0.85)

def print_items(keyword, all_cleaned_tweets, anaylsis):
    """ 
    function that prints out all the necessary information 
    including the polarity and the subjectivity of the specific keyword
    analysis refers to sentiment of tweets 
    """
    print()
    print("-----------------------------------------------------------------------------")
    print()
    print("The polarity and the subjectivity of the keyword" +" '"+ (keyword) +"' is below.")
    print(f"Polarity: {get_sentiment(all_cleaned_tweets)[0]:.3f}", f"Subjectivity: {get_sentiment(all_cleaned_tweets)[1]:.3f}")
    print("Based on the polarity, the overall sentiment of the keyword" + " '"+ (keyword)+"' is " + get_label(analysis, threshold =0) + ".")
    print()
    print("-----------------------------------------------------------------------------")



if __name__ == "__main__":
    main() 
    print("For your reference...")
    print("Polarity is a float within the range [-1.0, 1.0], -1.0 being completely negative and 1.0 being completely positive.")
    print("Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.")  

