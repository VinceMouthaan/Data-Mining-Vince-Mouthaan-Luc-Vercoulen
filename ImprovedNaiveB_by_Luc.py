#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:42:15 2023

@author: lucvercoulen
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing required packages
import pandas as pd
# old code was import pamdas as pd, idk what pamdas is but i am sure it is not a package in python. 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as skl

# Data cleaning
url = 'https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-VinceMouthaan/master/datasets/NB-reddit-hate-speech.csv'
rawDF = pd.read_csv(url)  # importing the data
# the sabotaged code was rawDF = pd.read_excel(url) this does not work because a csv file cannot red by a read.excel funtion                                             

# Making newDF
newDF = rawDF        # create the DF that will be used                 
  
newDF = newDF.drop(['id', 'response'], axis=1)  # remove unnecessary columns                              
# the old code was newDF = newDF.drop(['id', 'response'], axis=0) , axis in the drop funtion stands for 0= row and 1 = collumn. It is not possible to remove a collumn if axis = 0                         

newDF = newDF.fillna(False)                                             # hatespeech is False, non-hatespeech is True
newDF['hate_speech_idx'] = newDF['hate_speech_idx'] == False            # desired boolean column

# Data understanding
newDF['hate_speech_idx'].value_counts(normalize=True)                   # 77% is hatespeech

# old code: yeshate = ' '.join([Text for Text in newDF[newDF['hate_speech_idx'] == True]['text']]) it shoul be False because we want to know if there is hate speech. if we say == True we can the opposite result
yeshate = ' '.join([Text for Text in newDF[newDF['hate_speech_idx'] == False]['text']])
wordcloud_yeshate = WordCloud(background_color='black').generate(yeshate)
plt.imshow(wordcloud_yeshate)                                           # wordcloud for hatespeech

nohate = ' '.join([Text for Text in newDF[newDF['hate_speech_idx'] == True]['text']])
wordcloud_nohate = WordCloud(background_color='white').generate(nohate)

plt.imshow(wordcloud_nohate)                                            # wordcloud for non-hatespeech

# Data preparation
vectorizer = TfidfVectorizer()                                          # rename vectorizer function, no limit on the number of words
vectors = vectorizer.fit_transform(newDF.text)                          # generating the frequency vectors for all words in all comment sets
WordList = vectorizer.get_feature_names_out()                           # retrieving the words per vector
wordsDF = pd.DataFrame(vectors.toarray(), columns = WordList)           # turning the data into a legible dataframe

xTrain, xTest, yTrain, yTest = train_test_split(wordsDF, newDF.hate_speech_idx)     # generating training and test sets (standard setting is 75/25)

# Modelling
bayes = MultinomialNB()                                                 # renaming the naive bayes function
bayes.fit(xTrain, yTrain)                                               # training the model

yPred = bayes.predict(xTest)                                            # making a prediction for the test set
yTrue = yTest                                                           # the correct answer is yTest

# old code accuracy = skl.acuracy_score(yTrue, yPred) there was a typo in the word accuracy, if a typo is in this code the funtion 'accuracy_score' will not run.
accuracy = skl.accuracy_score(yTrue, yPred)                             # calculate the accuracy

matrix = skl.confusion_matrix(yTrue, yPred)                             # generate a confusion matrix
labelNames = pd.Series(['hatespeech', 'non-hatespeech'])                # these last two lines make the confusion matrix more legible
print(pd.DataFrame(matrix, columns = 'Prediction ' + labelNames, index = 'Reality ' + labelNames))

"""
After some testing it was concluded that reducing the number of hatespeech
comments only has a negative effect on the accuracy of the model, which is why
this step is no longer present in the code. 
In hindsight I should have probably saved it for the purpose of the excercise.
