# Logistical Regression for Sentiment Analysis of Amazon Reviews.
# Michael Rizig

import nltk # for stopwords list
import ssl # for initilizing stopwords list
import math # for implementing sigmoid
import numpy as np # for numpy array
import threading # for efficientcy
import time
from preprocess_dataset import preprocess # from our preprocess_dataset.py file
from logistical_regression import logistical_regression # from our logistical_regression.py file
import os
BOW = dict()
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
    for sentence in BOW_Vocab:
        for x in sentence.split(" "):
            if str(x) in BOW:
                BOW[str(x)] = BOW.get(x) +1
            else:
                BOW[str(x)] =1
    
    return data

def top1000():
    topwords = []
    for i in range(10):
        max_key = max(BOW, key=BOW.get)
        topwords.append(max_key)
        BOW.pop(max_key)
    return topwords
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

BOW_Vocab = []
def load_data():
    """Load data (positive words, negative words, training set and testing set)
    
    returns [positive_words,negative_words,train_set,test_set]"""
    try:
        # load positive words into array
        positive_words = nltk.corpus.opinion_lexicon.positive()  
        # load negative words into array
        negative_words = nltk.corpus.opinion_lexicon.negative()
        # load testing dataset into array of tuples [data:string, label:int]
        test = open("dataset/test_formatted.csv", encoding="utf8")
        test_set = []
        for line in test:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
            x[1] = x[1].lower()
            test_set.append([x[1],int(x[0])])
        
        # load training dataset into array of tuples [data:string, label:int]
        training = open("dataset/train_formatted.csv", encoding="utf8")
        train_set = []
        for line in training:
            x = line.split(",")
            x[1] = x[1].replace("\n","")
            x[1] = x[1].lower()
            if(int(x[0]) ==1):
                BOW_Vocab.append(x[1])
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
data = initilize()
poswordsdict = dict(zip(data[0],data[0]))
negwwordsdict = dict(zip(data[1],data[1]))
lock = threading.Lock()
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

    count=0
    for sample in dataset:
        


        lock.acquire()
        if training:
            out = open("dataset/training_features.csv",'a')
        else:
            out = open("dataset/testing_features.csv", 'a')
        x1,x2,x3,x4,x5 = extract1(sample[0])
        out.write(f"{x1},{x2},{x3},{x4},{x5},{sample[1]}\n")
        lock.release()
        count+=1
        if count%(len(dataset)/5)==0:
            print("Progress on thread ID ", threading.get_ident(), ": ", 100*(count/len(dataset)), "%")
    return
topwords = top1000()
def extract1(sample):
    # how many tokens appear in positive lexicon dict
    x1 = len([x for x in sample.split(" ") if poswordsdict.get(x,False)==x])    
    # how many tokens apppear in negative lexicon doct 
    x2 =len([x for x in sample.split(" ") if negwwordsdict.get(x,False)==x])
    # counts negations 
    x3 = 0
    # how many '?' tokens exist in sample
    x4 = sample.count("?")
    # how many key positive words appear in sample
    x5 = sample.count("love") + sample.count("amazing")  + sample.count("loved")+  + sample.count("great")
    # extract pairs of 2 words to count number of negations
    ngrams = extract_ngrams(sample,2)
    # count number of negations
    for n in ngrams:
        if negwwordsdict.get(n[0],False) or n[0] == "not" or n[0] == "dont"  or n[0] == "don't" or n[0] == "didn't" or n[0] == "didnt"  and poswordsdict.get(n[1],False):
            # negative negation 
            x3+=1
    return (x1,x2,x3,x4,x5)
def extract_ngrams(text, n):
  """Extracts n-grams from a given text."""
  tokens = nltk.word_tokenize(text)
  ngrams = list(nltk.ngrams(tokens, n))
  return ngrams
def load_features(training=True):
    """Load features from pre-extracted file after extract_features has been run"""
    output = []
    if training:
        file = open("dataset/training_features.csv")
       
    else:
        file = open("dataset/testing_features.csv")
    for line in file:
        line =line.split(",")
        x = [int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4]), int(line[5].replace("\n",""))]
        output.append(x)
    return output

def live_demo():
    while True:
        string = input("\nEnter a string to determine sentiment (enter quit to exit):  ")
        if string == "quit":
            return
        x1,x2,x3,x4,x5 = extract1(string)
        prediction = logreg.predict(np.array([x1,x2,x3,x4,x5]))
        match prediction:
            case 1:
                print("This comment was positive!\n")
            case -1:
                print("This comment was negative!")
            case 0:
                print("This comment was neutral!")
            case _:
                print("An error occured.")
        
# initilize features
try:
    f = open("dataset/training_features.csv")
    if f.readline() =="":
        raise FileNotFoundError
except FileNotFoundError:
    print("Extracting training features from train_fornatted.csv on 4 threads...")
    threads=[]

    q1= data[2][:37500]
    q2= data[2][37500:75000]
    q3 = data[2][75000:112500]
    q4 = data[2][112500:]
    x =threading.Thread(target=extract_features, args = [q1])
    x.start()
    threads.append(x)
    y =threading.Thread(target=extract_features, args = [q2])
    y.start()
    threads.append(y)
    z =threading.Thread(target=extract_features, args = [q3])
    z.start()
    threads.append(z)
    a =threading.Thread(target=extract_features, args = [q4])
    a.start()
    threads.append(a)
    for t in threads:
        t.join()
try: 
    f = open("dataset/testing_features.csv")
    if f.readline() =="":
        raise FileNotFoundError
except FileNotFoundError:
    print("Extracting testing features from test_formatted.csv on 4 threads...")
    threads=[]

    q1= data[3][:7500]
    q2= data[3][7500:15000]
    q3 = data[3][15000:22500]
    q4 = data[3][22500:]
    x =threading.Thread(target=extract_features, args = [q1,False])
    x.start()
    threads.append(x)
    y =threading.Thread(target=extract_features, args = [q2,False])
    y.start()
    threads.append(y)
    z =threading.Thread(target=extract_features, args = [q3,False])
    z.start()
    threads.append(z)
    a =threading.Thread(target=extract_features, args = [q4,False])
    a.start()
    threads.append(a)
    for t in threads:
        t.join()
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
#print(testing_features)
logreg = logistical_regression()
# fit (train) the model on our training set
print("Training...")
logreg.fit(training_features[:,[0,1,2,3,4]],training_features[:,5])
print("Testing...")
logreg.run(testing_features[:,[0,1,2,3,4]],testing_features[:,5])
#print(testing_features)
live_demo()