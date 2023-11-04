# -------------------------------------------------------------------------
# AUTHOR: Tommy James
# FILENAME: svm.py
# SPECIFICATION: Complete the Python program (svm.py) that will read the file optdigits.tra (3,823 samples)
# that includes training instances of handwritten digits (optically recognized). Read the file
# optdigits.names to get detailed information about this dataset. Also, check the file optdigits-orig.tra and
# optdigits-orig.names to see the original format of this data, and how it was transformed to speed-up the
# learning process (pre-processing phase). Your goal is to build multiple SVM classifiers using this data.
# You will simulate a grid search, trying to find which combination of four SVM hyperparameters (c,
# degree, kernel, and decision_function_shape) leads you to the best prediction performance. To test the
# accuracy of those distinct models, you will use the file optdigits.tes (1,797 samples).

# FOR: CS 4210- Assignment #3
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

# reading the training data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)

# getting the first 64 fields to create the feature training data and convert them to NumPy array
X_training = np.array(df.values)[:, :64]
# getting the last field to create the class training data and convert them to NumPy array
y_training = np.array(df.values)[:, -1]

# reading the training data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)

# getting the first 64 fields to create the feature testing data and convert them to NumPy array
X_test = np.array(df.values)[:, :64]
# getting the last field to create the class testing data and convert them to NumPy array
y_test = np.array(df.values)[:, -1]

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
# --> add your Python code here
max_acc = 0

for i in c:
    for j in degree:
        for k in kernel:
            for l in decision_function_shape:

                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                # For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                # --> add your Python code here

                clf = svm.SVC(C=i, degree=j, kernel=k,
                              decision_function_shape=l)

                # svm.SVC(C=i, degree=j, kernel=k, decision_function_shape=l)

                # Fit SVM to the training data
                # --> add your Python code here
                clf.fit(X_training, y_training)

                # make the SVM prediction for each test sample and start computing its accuracy
                # hint: to iterate over two collections simultaneously, use zip()
                # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                # to make a prediction do: clf.predict([x_testSample])
                # --> add your Python code here
                acc = 0

                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    y_pred = clf.predict([x_testSample])
                    acc = acc + accuracy_score([[y_testSample]], y_pred)
                    # print(f'{y_pred}\t{y_testSample}\t{acc}')

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                # with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                # --> add your Python code here
                accuracy = round((acc / len(X_test)), 2)

                if accuracy > max_acc:
                    max_acc = accuracy

                print(
                    f'Highest SVM accuracy so far: {max_acc}, Parameters: a = {i}, degree = {j}, kernel = {k}, decision_function_shape = {l}')
