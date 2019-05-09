from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
test_index = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)

# testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

# print real label from dataset
print(test_target)

# print predicted label from data test
print(classifier.predict(test_data))

# visualize the tree
tree.export_graphviz(classifier, out_file='tree.dot', filled=True,
                     feature_names=iris.feature_names, class_names=iris.target_names)

print(test_data[0], test_target[0])
print(iris.feature_names, iris.target_names)
