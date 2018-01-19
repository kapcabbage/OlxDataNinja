#MÓJ KOD
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





#CECHY TEKSTOWE ========================================================================
#Pobranie plików z cechami

cecha_descriptions = loadAttributeFileAsLines('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/titles/titles0')
cecha_desc_train, cecha_dest_test = splitLines(cecha_descriptions)

#Zamiana słów na sechy
count_vect_descriptions = CountVectorizer(stop_words=['a', 'i', 'o', 'na', 'w', 'po'], max_features = 1000)
X_train_counts = count_vect_descriptions.fit_transform(cecha_desc_train) #fit: robi listę słów i przydziela im indeksy#transform - zlicza listę słów
X_test_counts = count_vect_descriptions.transform(cecha_dest_test)

#Tfidf na cechach tekstowych
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) #dopasowanie do danych i przekształcenie macierzy
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

saver.dump(count_vect_descriptions, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/CV")
saver.dump(tfidf_transformer, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TFIDF")

saver.dump(X_train_tfidf, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/X_train_tekstowe_titles")
saver.dump(X_test_tfidf, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/X_test_tekstowe_titles")







#CECHY MANUALNE ========================================================================
cecha_hasTitle = loadAttributeFileAsLines('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/titles/titles0')


X_train = dodajCeche(X_train_tfidf, cecha_hasTitle)


saver.dump(X_train_tfidf, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/X_train");



#UCZENIE ===============================================================================

wynik_wasSold = loadAttributeFromFile('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0')

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, wynik_wasSold)



