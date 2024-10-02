# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list
import math # for implementing sigmoid
import threading # for efficientcy
import concurrent.futures # for thread management


def initilize():
    """Initilizees the model by downloading stopwords list, loading data, and preprocessing datasetsS"""
    download_stopwords()
    data = load_data()

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
        positive = open("dataset/positive.txt")
        positive_words = []
        for line in positive:
            line = line.replace('\n','')
            positive_words.append(line)
        
        # load negative words into array
        negative = open("dataset/negative.txt")
        negative_words = []
        for line in negative:
            line = line.replace('\n','')
            negative_words.append(line)
        
        # load testing dataset into array of tuples [data:string, label:int]
        test = open("dataset/test_amazon.csv")
        test_set = []
        for line in test:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
            test_set.append([x[1],int(x[0])])
        
        # load training dataset into array of tuples [data:string, label:int]
        training = open("dataset/train_amazon.csv")
        train_set = []
        for line in training:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
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



def extract_features():
    """ Takes in set of tokens and returns feature set
    
    output X = [x1,x2,x3,x4,x5]
    
    x1 = # of positive lexicons 

    x2 = # of negative lexicons

    x3 = if no ∈ sample (either 0 or 1 value)

    x4 = log(word count) 
    
    x5 = tbd"""

    pass

def sigmoid(x):
    """implements sigmoid function y = 1/ (1+e^-z)"""
    return 1 / (1 + math.e**(-x))
def logistical_regression(sample):
    """logreg function via sigmoid to return a value between 0-1
    representing the probability of a given class.
    If output > .5, it returns class 1,
    else output <= .5 and returns class 0.
    :input is a sample"""
    pass

data = initilize()

