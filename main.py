# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list
import math # for implementing sigmoid
import threading # for efficientcy
import concurrent.futures # for thread management

from preprocess_dataset import preprocess

def initilize():
    """Initilizees the model by downloading stopwords list, loading data, and preprocessing datasetsS"""
    download_stopwords()
    data = load_data()
    # if data is not preprocessed uncomment below (this will take time)
    # preprocess()
    # if features are not extracted uncomment below
    # extract_features(data[2])
    # extract_features(data[3], False)
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
            line = line.lower()
            positive_words.append(line)
        
        # load negative words into array
        negative = open("dataset/negative.txt")
        negative_words = []
        for line in negative:
            line = line.replace('\n','')
            line = line.lower()
            negative_words.append(line)
        
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

    x4 = ∃ '!' ∈ sample (either 0 or 1)

    x5 = log(word count) 
    
    c = class (1 = positive class 0 = negative class)
    """
    if training:
        out = open("dataset/training_features.csv",'w')
    else:
        out = open("dataset/testing_features.csv", 'w')
    for sample in dataset:
        x1 = len([x for x in sample[0].split(" ") if positive_words.count(x)>0])
        x2 =len([x for x in sample[0].split(" ") if negative_words.count(x)>0])
        x3 = 1 if sample[0].count("no")>0 else 0
        x4 = 1 if sample[0].count("!")>0 else 0
        x5 = math.log(len(sample[0].split(" ")),10)
        out.write(f"{x1},{x2},{x3},{x4},{x5},{sample[1]}\n")
        


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
positive_words  = data[0]
negative_words = data[1]
training_set = data[2]
testing_set = data[3]

extract_features(training_set)
print("half")
extract_features(testing_set, False)

