# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list
import math # for implementing sigmoid
def initilize():
    """ Download nessessary list of stopwords from nltk """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    return

def preprocess(textstring):
    """Preprocesses a string by removing stop words, symbols, digits, etc.
    outputs a list of tokens."""
    
    # assert that a string was passed
    assert isinstance(textstring,str) , "THIS IS NOT A STRING"
    # parse string into array of words
    words = textstring.split(" ")
    # remove stopwords
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    print(words)
    pass

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

initilize()
