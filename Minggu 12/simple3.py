from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


class OwnKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
        return self.y_train[best_index]


# Import Dataset
iris = load_iris()

# X as iris 4 features, y as iris label 0, 1, dan 2
X = iris.data
y = iris.target

# 10-fold Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

# use KNN Classifier
#from sklearn.neightbors import KNeighborsClassifier
classifier = OwnKNN()

# train the Classifier
classifier.fit(X_train, y_train)

# test the data
predictions = classifier.predict(X_test)

# calculate and display accuracy score
print(accuracy_score(y_test, predictions))
