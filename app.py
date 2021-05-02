from flask import Flask, request, jsonify
from flask_cors import CORS
import tweepy

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

import os
import re, string

# Setup flask application
app = Flask(__name__)
CORS(app)

# Setup tweepy
auth = tweepy.AppAuthHandler(os.environ.get("TWITTER_API_KEY"), os.environ.get("TWITTER_API_SECRET"))
api = tweepy.API(auth)

# Setup nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Setup transformers
classifier = pipeline('sentiment-analysis')
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = TFAutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

#import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

@app.route('/getAnalysis', methods=['POST'])
def getAnalysis():
    query = request.json.get("query")
    amount = int(request.json.get("amount", 50))
    data_type = int(request.json.get("type", 50))

    print(query)
    print(amount)
    print(data_type)

    if query:

        tweets = []
        general_analysis = {}
        pos = 0
        neu = 0
        neg = 0

        for tweet in tweepy.Cursor(api.search, q=query, lang="en", tweet_mode="extended").items(amount):
            if 'retweeted_status' in dir(tweet):
                text = tweet.retweeted_status.full_text
            else:
                text = tweet.full_text

            link = 'https://www.twitter.com/' + tweet.user.screen_name+'/status/' + tweet.id_str

        #     "compound": 0.6588,
        # "neg": 0.0,
        # "neu": 0.804,
        # "pos": 0.196

    # "label": "NEGATIVE",
    # "score": 0.9805542230606079

        #     0.005604513455182314, bad
        # 0.22520072758197784, netural
        # 0.7691947817802429 good

            if data_type == 0:
                analysis = sia.polarity_scores(preprocess(text))

                if analysis["compound"] >= 0.05:
                    classification = "positive"
                elif analysis["compound"] <= -0.05:
                    classification = "negative"
                else:
                    classification = "neutral"

                confidence = analysis["compound"] * 100
                analysis = {"classification": classification, "confidence": confidence}

            elif data_type == 1:
                pass
                analysis = classifier(preprocess(text))[0]
                classification = analysis["label"].lower()
                confidence = analysis["score"] * 100
                analysis = {"classification": classification, "confidence": confidence}
            else:
                pass
                encoded_input = tokenizer(preprocess(text), return_tensors='tf')
                output = model(encoded_input)
                scores = output[0][0].numpy()
                analysis = tf.nn.softmax(scores).numpy().tolist()

                max_score = max(analysis)
                max_index = analysis.index(max_score)

                if max_index == 0:
                   classification = "negative"
                elif max_index == 1:
                   classification = "neutral"
                else:
                   classification = "positive"

                confidence = max_score * 100

                analysis = {"classification": classification, "confidence": confidence}

            if analysis["classification"] == "positive":
                pos += 1
            elif analysis["classification"] == "neutral":
                neu += 1
            else:
                neg += 1

            tweets.append({"text": text, "link": link, "analysis": analysis})

        total = pos + neu + neg
        general_analysis = {
            "positive": pos/total * 100,
            "neutral": neu/total * 100,
            "negative": neg/total * 100
        }

        #import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        return jsonify({
            "message": f"Success! Your post request would include a query of {query} containing {amount} tweets",
            "tweets": tweets,
            "analysis": general_analysis
        }), 200
    else:
        return jsonify({
            "Message": "There was an error processing your request."
        }), 400


@app.route('/')
def index():
    return "<h1>All systems operational</h1>"


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
    # print("WE MADE THIS TWITTER SO WE DONT GO TO THE MOON ALONE! ONE FOR ALL ALL FOR ONE!")
    # print(sia.polarity_scores("WE MADE THIS TWITTER SO WE DONT GO TO THE MOON ALONE! ONE FOR ALL ALL FOR ONE!"))
    # print("WE MADE THIS TWITTER SO WE DONT GO TO THE MOON ALONE! ONE FOR ALL ALL FOR ONE!")
    # print(classifier('WE MADE THIS TWITTER SO WE DONT GO TO THE MOON ALONE! ONE FOR ALL ALL FOR ONE!'))

# git
# why heroku local instead of simply running
#positive sentiment: compound score >= 0.05
#neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#negative sentiment: compound score <= -0.05