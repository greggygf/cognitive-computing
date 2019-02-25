# sklearn implementation
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

path = str(
    '../')
data_training = np.genfromtxt(
    path+'IRIS-edit-training.csv', delimiter=',', dtype='unicode')
data_testing = np.genfromtxt(
    path+'IRIS-edit-testing.csv', delimiter=',', dtype='unicode')


def Preprocess(data):
    for singleData in data:
        if (singleData[4] == 'Iris-versicolor'):
            singleData[4] = 1
        else:
            singleData[4] = 0
    return data


data_training = Preprocess(data_training).astype(float)
data_testing = Preprocess(data_testing).astype(float)

x_training = data_training[:, 0:4]
y_training = data_training[:, 4]

x_test = data_testing[:, 0:4]
y_test = data_testing[:, 4]
print(y_test)

clf = Perceptron(random_state=None, eta0=0.1, shuffle=False,
                 penalty=None, class_weight=None, fit_intercept=False)

clf.fit(x_training, y_training)

y_predict = clf.predict(x_test)

print("sklearn perceptron coeffs:")
print(clf.coef_)

print("Akurasi sklearn Perceptron :")
print(accuracy_score(y_test, y_predict))
