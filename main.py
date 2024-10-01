# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list

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
def logisticalRegression(sample):
    """logreg function via sigmoid to return a value between 0-1
    representing the probability of a given class.
    If output > .5, it returns class 1,
    else output <= .5 and returns class 0.
    :input is a sample"""
    pass

initilize()
preprocess("Not worth the money: Banks'book Oscilloscope Guide uses large print and offers little information and there are mistakes...There is some useful information there but not worth the high price.")