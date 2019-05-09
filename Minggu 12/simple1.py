from sklearn import tree

features = [[200, 0], [180, 0], [130, 1], [142, 1]]
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)
print(classifier.predict([[150, 1]]))
