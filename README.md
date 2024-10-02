This repo holds implementation of a logistic regression algorithm in Python used to predict sentiment of reviews (either negative or positive). Classification, training, and testing using a real Amazon product review dataset.

Step by Step Guide:

<b>Step 1:</b><br> <br>The first step is data preprocessing. We will start preprocessing by removing stop words from our data.<br>
The preprocess() function reads our csv raw data and outputs 2 new files:<br>
-test_formatted.csv : test_amazon.csv with all stopwords removed<br>
-train_formatted.csv : train_amazon.csv with all stopwords removed<br><br>
notes:

- this function will only run if test_formatted.csv and train_formatted.csv do not exist
- ie this will only run the first time you run main.py
- you can run this portion of the program manually by running the preprocess_dataset.py script

<br><b>Step 2:</b><br><br> The second step is to extract features from our now preprocessed data.
The extract_features() function takes in both test_formatted.csv and train_formatted.csv and outputs 2 new files:<br>

-testing_features.csv : holds all features as well as class labels from testing dataset<br>
-training_features.csv : holds all features as well as class labels from training dataset
<br><br>
notes:

- a 'feature' set are described as:<br><br> X = [x1,x2,x3,x4,x5, c]
  <br>where:
  <br>x1 = # of positive lexicons
  <br>x2 = # of negative lexicons
  <br>x3 = if no ∈ sample (either 0 or 1 value)
  <br>x4 = ∃ '!' ∈ sample (either 0 or 1)
  <br>x5 = log(word count)
  <br>c = class (1 = positive class 0 = negative class)
- if testing_features.csv and training_features.csv already exist this function will not run <br>

<br><b>Step 3:</b><br><br> Next is to load features from our feature files.<br>
The load_features() function will run every time the model is run, and will load our features from training_features.csv and testing_features.csv into memory. <br>Below is an example of what a feature looks like: <br><br>
(Positive word count, Negative word count, "no" count, "!" count, log(word count), class)

- 2 , 0 , 0 , 1 , 1.1139433523068367 , 1

<br><b>Step 4:</b><br><br>
Its time for our model to learn the features. Firstly, we 'fit' the model to our features which is our training step.<br><br>
For our features set X and label set y, we call logistical_regression.fit(X,y):<br>
X = m x n<br>
m = # of data samples<br>
n = number of feature<br>
y = 1 x m matrix of correct labels for X<br><br>
fit() works by initilizing a weights to a (1 x n) matrix of zeros and bias to 0<br>
For n itterations:<br>

- We take the dot product of our sample and weights matricies, then add our bias (similar to linear regression).
  By applying sigmoid to this output, we are converting this to logisitcal regression, and we get an output value between 0 and 1<br>
- We take this output, determine the error (loss), and take the derivative of that to determine our grad_weight and grad_bias
- We then multiply these values with our learning rate and subtract them from our weights and bias to determine our new weights for the next itteration
- By the end of n itterations, we have 'optimized' our weights and biases based on slowly approaching our local minimum for our loss funciton

<br><b>Step 5:</b><br><br>
Final step is to run and generate our accuracy and confusion matrix:<br>
The logistical_regression.run() function takes in a sample and its expected output, and runs the predict on that sample. It then calculates the accuracy as a function of correct/total and calculates the confusion matrix, storing our true positive (TP), true negative (TN), false positive (FP), False Negative(FN) values.
