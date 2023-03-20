""""
Hey Luc, ik heb (tot nu toe) vijf dingen aangepast. 
Ik raad je wel aan om van boven naar beneden te werken.
Succes!
"""
# Importing required packages
import pamdas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as skl

# Data cleaning
url = 'https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-VinceMouthaan/master/datasets/NB-reddit-hate-speech.csv'
rawDF = pd.read_excel(url)                                              # importing the data

# Making newDF
newDF = rawDF                                                           # create the DF that will be used
newDF = newDF.drop(['id', 'response'], axis=0)                          # remove unnecessary columns

newDF = newDF.fillna(False)                                             # hatespeech is False, non-hatespeech is True
newDF['hate_speech_idx'] = newDF['hate_speech_idx'] == False            # desired boolean column

# Data understanding
newDF['hate_speech_idx'].value_counts(normalize=True)                   # 77% is hatespeech

yeshate = ' '.join([Text for Text in newDF[newDF['hate_speech_idx'] == True]['text']])
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

accuracy = skl.acuracy_score(yTrue, yPred)                             # calculate the accuracy

matrix = skl.confusion_matrix(yTrue, yPred)                             # generate a confusion matrix
labelNames = pd.Series(['hatespeech', 'non-hatespeech'])                # these last two lines make the confusion matrix more legible
print(pd.DataFrame(matrix, columns = 'Prediction ' + labelNames, index = 'Reality ' + labelNames))

"""
After some testing it was concluded that reducing the number of hatespeech
comments only has a negative effect on the accuracy of the model, which is why
this step is no longer present in the code. 

In hindsight I should have probably saved it for the purpose of the excercise.
"""