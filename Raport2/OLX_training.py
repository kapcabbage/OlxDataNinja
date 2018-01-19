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


maxWierszy = 10000

#OLX_cechy = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_CECHY")
cechy_treningowe = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TRAIN_FULL").tocsr()[:maxWierszy,:]



lenCech = cechy_treningowe.shape[0]

# wynik_views = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_views")[:lenCech]
# wynik_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies")[:lenCech]
wynik_wasSold = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0")[:lenCech]


from sklearn.ensemble import RandomForestClassifier
klasyfikator = RandomForestClassifier(n_estimators=5)

# klasyfikator_views = klasyfikator.fit(cechy_treningowe, wynik_views)
# klasyfikator_replies = klasyfikator.fit(cechy_treningowe, wynik_replies)
klasyfikator_wasSold = klasyfikator.fit(cechy_treningowe, wynik_wasSold)


# saveObject(klasyfikator_views, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/KLASYFIKATOR_VIEWS")
# saveObject(klasyfikator_replies, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/KLASYFIKATOR_REPLIES")
saveObject(klasyfikator_wasSold, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/KLASYFIKATOR_WASSOLD")

