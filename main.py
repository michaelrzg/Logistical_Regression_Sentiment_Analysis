# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list
import math # for implementing sigmoid
import numpy as np # for numpy array
import threading # for efficientcy
import concurrent.futures # for thread management

from preprocess_dataset import preprocess # from our preprocess_dataset.py file
from logistical_regression import logistical_regression # from our logistical_regression.py file

def initilize():
    """Initilizees the model by downloading stopwords list, loading data, and preprocessing datasetsS"""
    #download list of stopwords from nltk
    #download_stopwords()

    # if data is not preprocessed, preprocess() will run
    try:
        file = open("dataset/test_formatted.csv")
    except FileNotFoundError:
        print("Preprocessing...")
        preprocesser = preprocess()
        preprocesser.preprocess_dataset()
    
    # load preprocessed data
    data = load_data()
    # if features not extracted, run extract features()
    

    
    return data


def download_stopwords():
    """ Download nessessary list of stopwords from nltk """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    return


def load_data():
    """Load data (positive words, negative words, training set and testing set)
    
    returns [positive_words,negative_words,train_set,test_set]"""
    try:
        # load positive words into array
        positive_words = nltk.corpus.opinion_lexicon.positive()  
        # load negative words into array
        negative_words = nltk.corpus.opinion_lexicon.negative()
        # load testing dataset into array of tuples [data:string, label:int]
        test = open("dataset/test_formatted.csv")
        test_set = []
        for line in test:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
            x[1] = x[1].lower()
            test_set.append([x[1],int(x[0])])
        
        # load training dataset into array of tuples [data:string, label:int]
        training = open("dataset/train_formatted.csv")
        train_set = []
        for line in training:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
            x[1] = x[1].lower()
            train_set.append([x[1],int(x[0])])
    except FileNotFoundError:
        print("one or more files not found. Check paths")
    
    return [positive_words,negative_words,train_set,test_set]


def remove_stopwords(textstring, outputs,i):
    """Preprocesses a string by removing stop words, symbols, digits, etc.
    outputs a list of tokens."""
    
    # assert that a string was passed
    assert isinstance(textstring[0],str) , "THIS IS NOT A STRING"
    # parse string into array of words
    words = textstring.split(" ")
    # remove stopwords
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    outputs.append([i,words])
    return


def extract_features(dataset, training=True):
    """ Takes in set of tokens and returns feature set
    
    output X = [x1,x2,x3,x4,x5, c]
    
    x1 = # of positive lexicons 

    x2 = # of negative lexicons

    x3 = if no ∈ sample (either 0 or 1 value)

    x4 = count("!")

    x5 = log(word count) 
    
    c = class (1 = positive class 0 = negative class)
    """
    if training:
        out = open("dataset/training_features.csv",'w')
        print("Extracting training set features...")
    else:
        out = open("dataset/testing_features.csv", 'w')
        print("Extracting testing set features...")
    count=0
    for sample in dataset:
        x1 = len([x for x in sample[0].split(" ") if data[0].count(x)>0])
        x2 =len([x for x in sample[0].split(" ") if data[1].count(x)>0])
        x3 = 1 if sample[0].count("not")>0 else 0
        x4 = sample[0].count("!")
        x5 = len(sample[0].split(" "))
        out.write(f"{x1},{x2},{x3},{x4},{x5},{sample[1]}\n")
        count = count+1
        print(count/len(dataset))
    return

def load_features(training=True):
    """Load features from pre-extracted file after extract_features has been run"""
    output = []
    if training:
        file = open("dataset/training_features.csv")
       
    else:
        file = open("dataset/testing_features.csv")
    for line in file:
        line =line.split(",")
        x = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4]), int(line[5].replace("\n",""))]
        output.append(x)
    return output

# initilize data
data = initilize()
try:
    file = open("dataset/testing_features.csv")
except FileNotFoundError:
    extract_features(data[2])
    extract_features(data[3], False)
# store appropriately
positive_words  = data[0]
negative_words = data[1]
training_set = data[2]
testing_set = data[3]

# load features from training data and convert to numpy array
training_features = load_features()
training_features = np.array(training_features)

# load features from testing data and convert to numpy array
testing_features = load_features(False)
testing_features = np.array(testing_features)
#create logistical regression model object form logistical_regression.py
print(testing_features)
logreg = logistical_regression()
# fit (train) the model on our training set
print("Training...")
logreg.fit(training_features[:,[0,1,2,3,4]],training_features[:,5])
print("Testing...")
logreg.run(testing_features[:,[0,1,2,3,4]],testing_features[:,5])
#print(testing_features)