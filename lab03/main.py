import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

def classify(classifier, test_inputs, test_classes):
    predicted_classes = classifier.predict(test_inputs)
    good_predictions = 0

    for i in range(len(test_classes)):
        if predicted_classes[i] == test_classes[i]:
            good_predictions +=1

    accuracy = accuracy_score(test_classes, predicted_classes)
    print(good_predictions)
    print(accuracy*100, "%")
    cm = confusion_matrix(test_classes, predicted_classes)
    print(cm)


df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=13)

print(test_set)
print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

print(test_inputs)
print(test_classes)

def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return "Setosa"
    elif sw > 3:
        return "Virginica"
    else:
        return "Versicolor"
    
print('My classifier:')
    
good_predictions = 0
test_len = test_set.shape[0]

for i in range(test_len):
    row = test_set[i]
    if classify_iris(row[0], row[1], row[2],
                     row[3]) == test_set[i][4]:
        good_predictions += 1

print(good_predictions)
print(good_predictions/test_len*100, "%")

print('\nDecision tree:')

clf_classifier = tree.DecisionTreeClassifier()
clf_classifier.fit(train_inputs, train_classes)

# plt.figure(figsize=(20,10))
# tree.plot_tree(clf, filled=True, feature_names=df.columns[:-1], class_names=df['variety'].unique())
# plt.show()

classify(clf_classifier, test_inputs, test_classes)

print('\nKNeighbors:')

ks = [3, 5, 11]

for k in ks:
    print(f'{k}:')
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(train_inputs, train_classes)
    classify(knn_classifier, test_inputs, test_classes)

print('\nNaive Bayes:')
nb_classifier = GaussianNB()
nb_classifier.fit(train_inputs, train_classes)
classify(nb_classifier, test_inputs, test_classes)
