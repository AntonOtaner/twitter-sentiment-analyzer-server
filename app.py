"""
Anton Otaner , 1930028
Friday , May 7
R. Vincent , instructor
Final Project
"""

from flask import Flask, request, jsonify  # web framework to handle requests for API
from flask_cors import CORS  # package to allow CORS with flask (and to allow the frontend to properly connect to the server)

import tweepy  # package to communicate with Twitter API

import nltk  # Natural Language Toolkit used for VADER model
from nltk.sentiment import SentimentIntensityAnalyzer  # Pretrained model with VADER

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # Natural Language Processing package with many different models
import torch  # needed to fix a problem from PythonAnywhere
from scipy.special import softmax  # needed to compute softmax function

import os  # used to get local environment variables

# Must set threads to one for package to work properly on server
torch.set_num_threads(1)

# Setup flask application
app = Flask(__name__)
CORS(app)  # setup CORS to have proper communication between frontend and server

# Setup tweepy
auth = tweepy.AppAuthHandler(os.environ.get("TWITTER_API_KEY"), os.environ.get("TWITTER_API_SECRET"))  # use local environment variables
api = tweepy.API(auth)

# Setup nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Setup transformers
classifier = pipeline('sentiment-analysis')  # defaults to BERT classifier
# get roBERTa Twitter model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


def preprocess(text):
    """
    Function to process text suitable for twitter roBERTa model. Function from https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment.
    Input string.
    Returns string (processes version of input).
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


@app.route('/getAnalysis', methods=['POST'])
def getAnalysis():
    """
    Function to analysis tweets based on query, amount and type. Will return a json with general analysis, and list tweets which each have their own analysis.
    """
    try:
        query = request.json.get("query")  # get search query from request
        amount = int(request.json.get("amount"))  # get amount of tweets to analyse from request
        data_type = int(request.json.get("type", 0))  # get type of model to anlayze tweets with from request and default it to type 0 as that is the default type on the frontend

        if query and amount:
            tweets = []  # will contain a list of tweets and their analysis
            pos = 0  # number of positive tweets
            neu = 0  # number of neutral tweets
            neg = 0   # number of negative tweets

            # get X amount of tweets from search query and loop through them
            # tweet mode is extended to get tweets in their full length
            for tweet in tweepy.Cursor(api.search, q=query, lang="en", tweet_mode="extended").items(amount):
                # check if tweet is retweeted to make sure to get all of the tweet text content
                if 'retweeted_status' in dir(tweet):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text

                # get link of tweet
                link = 'https://www.twitter.com/' + tweet.user.screen_name+'/status/' + tweet.id_str

                # get scores and analysis depending on model type
                if data_type == 0:
                    # analyse tweet with VADER
                    analysis = sia.polarity_scores(preprocess(text))

                    # Get positive, negative and neutral scores, as well as difference between postive and negative scores
                    negative_score = analysis["neg"]
                    neutral_score = analysis["neu"]
                    positive_score = analysis["pos"]
                    score_delta = abs(positive_score-negative_score)

                elif data_type == 1:
                    # analyse tweet with BERT
                    analysis = classifier(preprocess(text))[0]

                    # get classification and confidence
                    classification = analysis["label"].lower()
                    confidence = analysis["score"] * 100

                else:
                    # analyse tweet with roBERTa Twitter from https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
                    encoded_input = tokenizer(preprocess(text), return_tensors='pt')
                    output = model(**encoded_input)
                    scores = output[0][0].detach().numpy()
                    analysis = softmax(scores).tolist()  # compute softmax of score

                    # Get positive, negative and neutral scores, as well as difference between postive and negative scores
                    negative_score = analysis[0]
                    neutral_score = analysis[1]
                    positive_score = analysis[2]
                    score_delta = abs(positive_score-negative_score)

                # get classification and compute confidence for model type 0 and 2
                if data_type == 0 or data_type == 2:
                    # neutral is neutral rating is above 0.8, or if difference between positive and negative is less os equal to 0.05
                    if neutral_score > 0.8 or score_delta <= 0.05:
                        classification = "neutral"
                        score = 1 - score_delta * 2
                    elif positive_score > negative_score:  # positive if positive score is bigger than negative score
                        classification = "positive"
                        score = neutral_score / 2 + positive_score
                    else:  # negative if negative score is bigger than positive score
                        classification = "negative"
                        score = neutral_score / 2 + negative_score
                    confidence = score * 100

                analysis = {"classification": classification, "confidence": confidence}  # analysis for specific tweet

                # increment count of positive, negative or neutral tweets
                if analysis["classification"] == "positive":
                    pos += 1
                elif analysis["classification"] == "neutral":
                    neu += 1
                else:
                    neg += 1

                # add tweet content and analysis to tweet list
                tweets.append({"text": text, "link": link, "analysis": analysis})

            # if tweets were found for the search query
            tweet_amount = len(tweets)  # amount of tweets analyzed
            if tweet_amount > 0:
                # compute proportions of positive, negative and neutral tweets
                general_analysis = {
                    "positive": pos/tweet_amount * 100,
                    "neutral": neu/tweet_amount * 100,
                    "negative": neg/tweet_amount * 100
                }
                message = f"Success! Your post request would include a query of {query} containing {amount} tweets"  # success message
            else:
                message = f"There were no tweets found for a query of {query}"  # error message
                general_analysis = {}

            # send response with message, tweets, and the general analysis
            return jsonify({
                "message": message,
                "tweets": tweets,
                "analysis": general_analysis
            }), 200
        else:
            # If missing the query or the amount of tweets to analyse, send a response indicating there were missing inputs.
            return jsonify({
                "Message": "There was an error processing the request."
            }), 400
    except Exception as e:
        # If there is an error, printed out and send a response indicating there was an error.
        print(e)
        return jsonify({
            "Message": "There was an error getting the response."
        }), 500


@app.route('/')
def index():
    """
    Function to test if API is working. When post/get to server url, you will receive a confirmation if the server is working.
    """
    return "<h1>All systems operational</h1>"


if __name__ == '__main__':
    # Run the app on port 5000
    app.run(threaded=True, port=5000)
