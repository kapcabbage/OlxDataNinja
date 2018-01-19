import matplotlib.pyplot as plt
from sklearn.externals import joblib as saver
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import scipy.sparse as sparse
import sys

def loadAttributeFileAsLines(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def linesToCSRMatrix(lines):
    row = np.arange(len(lines))
    col = np.zeros(len(lines))
    data = np.array(lines).astype(float)
    return sparse.csr_matrix((data, (row, col)))

def loadAttributeFromFile(filename):
    return linesToCSRMatrix(loadAttributeFileAsLines(filename))


def dodajCeche(macierz, nowaCecha):
    return sparse.hstack((macierz, nowaCecha))


def saveObject(object, filename):
    saver.dump(object, filename)

def loadObject(filename):
    return saver.load(filename)

def splitLines(lines):
    cutPoint = len(lines) - int(len(lines)/5)
    return lines[:cutPoint], lines[cutPoint:]


inputs_train = 8000
inputs_test  = 2000
classes_amount = 2004

X_train = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TRAIN_BIGGER")[:inputs_train]
X_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TEST_BIGGER")[:inputs_test]

# lines_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies")
lines_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0")
wynik_replies_train, wynik_replies_test = splitLines(lines_replies)
wynik_replies_train = list(map(int, wynik_replies_train))[:inputs_train]
wynik_replies_test = list(map(int, wynik_replies_test))[inputs_train:inputs_train+inputs_test]





#REGRESJA LOGISTYCZNA
# from sklearn import linear_model
#
# print("LOGISTIC REGRESSION")
# klasyfikator = linear_model.LogisticRegression()
#
# klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_views_train)
#
# predicted = klasyfikator_wasSold.predict(X_test)
# roundedPredictedList = [ round(elem, 0) for elem in predicted ]
# roundedPredicted = list(map(int, roundedPredictedList))
# Y_test = wynik_views_test
#
# poprawnosc = np.mean(abs(predicted - Y_test) < 3)
# print(poprawnosc)

#NEIGHBOURS
# from sklearn.neighbors import KNeighborsClassifier
#
# print("NEIGHBORS")
# print("n_neighbors | Poprawność")
# for i in range(1, 10):
#     klasyfikator = KNeighborsClassifier(n_neighbors= 5, algorithm='kd_tree')
#
#     klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_views_train)
#
#     predicted = klasyfikator_wasSold.predict(X_test)
#     roundedPredictedList = [ round(elem, 0) for elem in predicted ]
#     roundedPredicted = list(map(int, roundedPredictedList))
#     Y_test = wynik_views_test
#
#     poprawnosc = np.mean(abs(predicted - Y_test) < 3)
#     statStr = str(round(i,1)) + " | " + str(poprawnosc)
#     print(statStr)


#BAGGING CLASIFFIER
# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# print("BaggingClassifier")
# print("MaxSamples | MaxFeatures | Poprawność")
# for i in range(1, 10):
#     for j in range(1, 10):
#         maxSamples = 0.1 * float(i)
#         maxFeatures= 0.1 * float(j)
#         klasyfikator = BaggingClassifier(KNeighborsClassifier(), max_samples=maxSamples, max_features=maxFeatures)
#
#         klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_replies_train)
#
#         predicted = klasyfikator_wasSold.predict(X_test)
#         roundedPredictedList = [ round(elem, 0) for elem in predicted ]
#         roundedPredicted = list(map(int, roundedPredictedList))
#         Y_test = wynik_replies_test
#
#         poprawnosc = np.mean(predicted == Y_test)
#         statStr = str(round(maxSamples,1)) + " | " + str(round(maxFeatures, 1)) + " | " + str(poprawnosc)
#         print(statStr)
#
# ia = 9
#DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
#
# print("DECISION TREE")
# print("min_samples_split | Poprawność")
# for i in range(2, 10):
#     klasyfikator = DecisionTreeClassifier(min_samples_split = i)
#
#     klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_views_train)
#
#     predicted = klasyfikator_wasSold.predict(X_test)
#     roundedPredictedList = [ round(elem, 0) for elem in predicted ]
#     roundedPredicted = list(map(int, roundedPredictedList))
#     Y_test = wynik_views_test
#
#     poprawnosc = np.mean(abs(predicted - Y_test) < 3)
#     statStr = str(round(i,1)) + " | " + str(poprawnosc)
#     print(statStr)



#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

print("RANDOM FOREST")
print("n_estimators | Poprawność")

wInd = np.arange(classes_amount)
weight_list= {}
for i in range(len(wInd)):
    weight_list[wInd[i]] = 1


weight_list[len(weight_list)-1] = 70
weight_list[len(weight_list)-2] = 70
weight_list[len(weight_list)-3] = 50
weight_list[len(weight_list)-4] = 50

klasyfikator = RandomForestClassifier(n_estimators=5)

klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_replies_train )

predicted = klasyfikator_wasSold.predict(X_test)
roundedPredictedList = [ round(elem, 0) for elem in predicted ]
roundedPredicted = list(map(int, roundedPredictedList))
Y_test = wynik_replies_test

poprawnosc = np.mean(predicted == Y_test)
statStr = str(round(5,1)) + " | " + str(poprawnosc)
print(statStr)


#EXTRA TREEES
# from sklearn.ensemble import ExtraTreesClassifier
#
# print("EXTRA TREEES")
# print("n_estimators | Poprawność")
# for i in range(1, 10):
#     klasyfikator = ExtraTreesClassifier(n_estimators = i, min_samples_split = 4)
#
#     klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_views_train)
#
#     predicted = klasyfikator_wasSold.predict(X_test)
#     roundedPredictedList = [ round(elem, 0) for elem in predicted ]
#     roundedPredicted = list(map(int, roundedPredictedList))
#     Y_test = wynik_views_test
#
#     poprawnosc = np.mean(abs(predicted - Y_test) < 3)
#     statStr = str(round(i,1)) + " | " + str(poprawnosc)
#     print(statStr)


plt.plot(np.subtract(wynik_views_test, predicted))

a=9