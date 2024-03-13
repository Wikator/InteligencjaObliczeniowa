import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * -0.46122 + waga * 0.97314 + wzrost * -0.39203 + 0.80109
    hidden2 = wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847 + 0.43529
    activasion = lambda x: 1/(1 + math.e**-x)
    return activasion(hidden1) * -0.81546 + activasion(hidden2) * 1.03775 - 0.2368

print(forwardPass(23, 75, 176))
print(forwardPass(25, 67, 180))
print(forwardPass(28, 20, 175))
print(forwardPass(22, 65, 165))


df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=13)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(3,3,),
                    random_state=1)

clf.fit(train_inputs, train_classes)

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

classify(clf, test_inputs, test_classes)

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=13)

train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,6,3,2,),
                    random_state=1,
                    activation='identity',
                    max_iter=500)

clf.fit(train_inputs, train_classes)

classify(clf, test_inputs, test_classes)

# False negative w tym przypadku są dużo gorsze
# W tym wytrenowanym modelu jest 15 FP i 34 FN


