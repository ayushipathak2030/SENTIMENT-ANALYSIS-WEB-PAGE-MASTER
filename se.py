import nltk
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber


import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request 
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
train = [
   ('I love this sandwich.', 'pos'),
   ('This is an amazing place!', 'pos'),
   ('I feel very good about these beers.', 'pos'),
   ('I do not like this restaurant', 'neg'),
   ('I am tired of this stuff.', 'neg'),
   ("I can't deal with this", 'neg'),
   ("My boss is horrible.", "neg")
 ]
cl = NaiveBayesClassifier(train)
cl.classify("I feel amazing!")


#read the file
data = pd.read_csv('C:/Program Files/spot.csv', encoding="utf-8")

# replace 4 with 'post' and 0 as 'neg' in 'polarity' column
data['polarity'] = data.replace({1: 'post', 9: 'neg' , 0:'neu'})

# convert the data into a list
data = data[['polarity', 'review']].values.tolist()


L = len(data)
train_index = int(0.60 * L)

# split the data into a train and test data
train, test = data[:train_index], data[train_index: ]



cl = NaiveBayesClassifier(train)
print(cl.accuracy(test))


blob = TextBlob("The beer is very good.", classifier=cl)
a=blob.sentiment.polarity
print(a)




def get_sentiment(text):
    blob=TextBlob(text,classifier=cl)
    score=blob.sentiment.polarity
    if score > 0:
        if score > 0.5:
            return "This sentence is highly positive."
        else:
            return "This sentence is positive."
    elif score == 0:
        return "This sentence is neutral."
    else:
        if score < -0.5:
            return "This sentence is highly negative." # -0.9
        else:
            return "This sentence is negative." # -0.3


app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        content = request.form.get("text")
        sentiment = get_sentiment(content)
        return "The sentiment is: " + sentiment
    return render_template("new 1.html")


   

app.run(debug= True)
