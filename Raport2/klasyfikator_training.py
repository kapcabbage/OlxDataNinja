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


#WYNIK : wasSold
lines_wasSold = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0")
wynik_wasSold_train, wynik_wasSold_test = splitLines(lines_wasSold)
wynik_wasSold_train = list(map(int, wynik_wasSold_train))
wynik_wasSold_test = list(map(int, wynik_wasSold_test))

#WYNIK : replies
lines_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies")
wynik_replies_train, wynik_replies_test = splitLines(lines_replies)
wynik_replies_train = list(map(int, wynik_replies_train))
wynik_replies_test = list(map(int, wynik_replies_test))

#WYNIK : views
lines_views = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_views")
wynik_views_train, wynik_views_test = splitLines(lines_views)
wynik_views_train = list(map(int, wynik_views_train))
wynik_views_test = list(map(int, wynik_views_test))


#CECHA : hasPicture
lines_hasPicture = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/pictures/hasPicture0")
lines_hasPicture_train, lines_hasPicture_test = splitLines(lines_hasPicture)

cecha_hasPicture_train = linesToCSRMatrix(lines_hasPicture_train)
cecha_hasPicture_test = linesToCSRMatrix(lines_hasPicture_test)

#CECHA : age
lines_age = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_age")
lines_age_train, lines_age_test = splitLines(lines_age)

cecha_age_train = linesToCSRMatrix(lines_age_train)
cecha_age_test = linesToCSRMatrix(lines_age_test)

#CECHA : hasNumber_description
lines_hasNumber_description = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_hasNumber_description")
lines_hasNumber_description_train, lines_hasNumber_description_test = splitLines(lines_hasNumber_description)

cecha_hasNumber_description_train = linesToCSRMatrix(lines_hasNumber_description_train)
cecha_hasNumber_description_test = linesToCSRMatrix(lines_hasNumber_description_test)

#CECHA : hasNumber_title
lines_hasNumber_title = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_hasNumber_title")
lines_hasNumber_title_train, lines_hasNumber_title_test = splitLines(lines_hasNumber_title)

cecha_hasNumber_title_train = linesToCSRMatrix(lines_hasNumber_title_train)
cecha_hasNumber_title_test = linesToCSRMatrix(lines_hasNumber_title_test)

#CECHY TEKSTOWE : TRENINGOWE
cecha_decritption_train = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_train_tekstowe")
cecha_titles_train = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_train_tekstowe_titles")

cecha_text_train = dodajCeche(cecha_decritption_train, cecha_titles_train)

#CECHY TEKSTOWE : TESTOWE
cecha_decritption_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_test_tekstowe")
cecha_titles_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_test_tekstowe_titles")

cecha_text_test = dodajCeche(cecha_decritption_test, cecha_titles_test)


#==========================================
#LĄCZYMY CECHY
X_train = cecha_text_train
# X_train = X_train.tocsr()[:,:2000]
X_train = dodajCeche(X_train, cecha_hasPicture_train)
X_train = dodajCeche(X_train, cecha_age_train)
X_train = dodajCeche(X_train, cecha_hasNumber_description_train)
X_train = dodajCeche(X_train, cecha_hasNumber_title_train)
# X_train = X_train.tocsr()[:8000,:]
saveObject(X_train, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TRAIN_FULL")

# X_test = cecha_text_test
# X_test = dodajCeche(X_test, cecha_hasPicture_test)
# X_test = dodajCeche(X_test, cecha_age_test)
# X_test = dodajCeche(X_test, cecha_hasNumber_description_test)
# X_test = dodajCeche(X_test, cecha_hasNumber_title_test)
# saveObject(X_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TEST_FULL")

# saveObject(wynik_views_train, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/WYNIK_VIEWS_TRAIN")
# saveObject(wynik_views_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/WYNIK_VIEWS_TEST")
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier



#Regresja Liniowa
#klasyfikator = linear_model.LinearRegression()
from sklearn.ensemble import RandomForestClassifier
#klasyfikator = RandomForestClassifier(n_estimators=2)

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
klasyfikator = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)



klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_views_train)

predicted = klasyfikator_wasSold.predict(X_test.tocsr()[:200,:])
roundedPredictedList = [ round(elem, 0) for elem in predicted ]
roundedPredicted = list(map(int, roundedPredictedList))
Y_test = wynik_views_test[:200]

poprawnosc = np.mean(abs(predicted - Y_test) < 3)

print("Regresja liniowa: ")
print(poprawnosc)


plt.plot(np.subtract(wynik_wasSold_test, predicted))


#Regresja Logistyczna
#klasyfikator = linear_model.LogisticRegression()

klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_wasSold_train) #trenowanie klasyfikatora

predicted = klasyfikator_wasSold.predict(X_test)
poprawnosc = np.mean(predicted == wynik_wasSold_test)
print("Regresja logistyczna: ")
print(poprawnosc)

# Y = np.arrange(200000)
# Y = [x for pair in zip(Y,Y) for x in pair]




# #Naiwny Bayes
# klasyfikator_wasSold = MultinomialNB().fit(X_train, wynik_wasSold_train) #trenowanie klasyfikatora
#
# predicted = klasyfikator_wasSold.predict(X_test)
# poprawnosc = np.average(predicted == wynik_wasSold_test)
# print("Naiwny Bayes: ")
# print(poprawnosc)


# #Metoda Wektorów Nośnych
# klasyfikator = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
#
# klasyfikator_wasSold = klasyfikator.fit(X_train, wynik_wasSold_train) #trenowanie klasyfikatora
#
# predicted = klasyfikator_wasSold.predict(X_test)
# poprawnosc = np.average(predicted == wynik_wasSold_test)
# print("Metoda Wektorów Nośnych: ")
# print(poprawnosc)