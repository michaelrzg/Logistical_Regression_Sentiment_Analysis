# logistical regression model
# michael rizig
import numpy as np
import math

class logistical_regression():
    """Logistical Regression Model. Run .predict() to return a value between 0-1
    representing the probability of a given class. If output > .5, it returns class 1,
    else output <= .5 and returns class 0."""
    #constructor
    def __init__(self, learning_rate=.01,itterations=5000) -> None:
        self.lr = learning_rate
        self.itterations = itterations
        self.weights = None
        self.bias = None


    def sigmoid(self,x):
        """implements sigmoid function y = 1/ (1+e^-z)"""
        return 1 / (1 + np.exp(-x))
    
    def fit(self, sample_X, desired_attribute_y):
        """This function is our 'training' step. 
        By taking in features X and desires output y, we fit a model around this data.
        """
        # .shape returns # of rows, # of columns
        n_samples, n_features = sample_X.shape
        #initilize our weights to 0
        self.weights = np.zeros(n_features)
        #initilize our bias to 0
        self.bias = 0
        # training loop
        for i in range(self.itterations):
            # prediction =
            # features matrix X * weights w + bias
            # we then take the sigmoid of this to return us the value between 0 and 1
            pred = self.sigmoid(np.dot(sample_X,self.weights) + self.bias)
           

            # calculating error = taking derivative of loss
            # formula :
            # 1/N * dotproduct(X, difference between predictions and actual)
            # X needs to be transposed since X = (# of sample x number of features) and y = (num of samples x 1)
            # we need to make x = (m*n) and y = n*l so we take transpose of X
            grad_weights = (1/n_samples) * np.dot(sample_X.T,(pred-desired_attribute_y))
            grad_bias = (1/n_samples) * np.sum(pred-desired_attribute_y)
            
            # finally we update our weights and biases to new values for next loop
            # weights = (weights - ( lr * gradient of weights)
            # we are now "decending" towards local minimum of loss function
            self.weights = self.weights  - (self.lr * grad_weights)
            self.bias = self.bias - (self.lr * grad_bias)

    def predict(self, sample):
        """Use our trained weights and bias to predict a novel set and return set of labels"""

        # to predict a given sample, we run the same prediction as before but now with our trained weights
        pred = self.sigmoid(np.dot(sample,self.weights) + self.bias)

        return [1 if x>.5 else 0 for x in pred]
    def run(self, sample, expected):
        """Use our trained weights and bias to predict a novel set, then calculates accuracy and confusion matrix"""
        # run pred and get output list of labels
        pred = self.predict(sample)
        # store true positive (TP), true negative (TN), false positive (FP), False Negative(FN)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        correct=0
        for i in range(len(pred)):
            if(pred[i] == 0 and pred[i] == expected[i]):
                TN+=1
                correct+=1
            elif(pred[i] ==1 and pred[i] == expected[i]):
                TP+=1
                correct+=1
            elif(pred[i] ==0 and pred[i] != expected[i]):
                FN +=1
            else:
                FP +=1
        print("Accuracy: ", correct/len(pred),
              "\n Confuion Matrix:",
              "\n   +        -"
              "\n+ ",TP,"  ", FP,
              "\n- ",FP, "  ", FN)
                




