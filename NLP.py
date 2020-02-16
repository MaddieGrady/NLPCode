#import modules and functions
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

#Import news article dataset from sklearn
from sklearn.datasets import fetch_20newsgroups

#Load training and test data
news_train = fetch_20newsgroups(subset='train', shuffle=True)
news_test = fetch_20newsgroups(subset='test', shuffle=True)

#build pipeline to prepare data for classification;
#Take features from text, Bag of Words model using count vector, remove stop words.
#transfrom so longer documents dont have more weight reduce the weight of very common words (e.g. the)
classifier = Pipeline([('vect', CountVectorizer(stop_words='english')),
                   ('tfidf', TfidfTransformer()),
                   ('classifier', MultinomialNB()),
                      ])
#fit classifier
classifier.fit(news_train.data, news_train.target)

#predict on test data
predicted = classifier.predict(news_test.data)

#measure accuracy
accuracy = np.mean(predicted == news_test.target)
print(accuracy)

