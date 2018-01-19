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

# lines_views = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_views")
# lines_views_train, lines_views_test = splitLines(lines_views)
# wynik_views_test = list(map(int, lines_views_test))
# saveObject(wynik_views_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_views_test")

# lines_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies")
# lines_replies_train, lines_replies_test = splitLines(lines_replies)
# wynik_replies_test = list(map(int, lines_replies_test))
# saveObject(wynik_replies_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_replies_test")
#
# lines_wasSold = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0")
# wynik_wasSold_train, wynik_wasSold_test = splitLines(lines_wasSold)
# wynik_wasSold_test = list(map(int, wynik_wasSold_test))
# saveObject(wynik_wasSold_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_wasSold_test")


elementsCount = 1000

wynik_views_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_views_test")[:elementsCount]
# wynik_replies_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_replies_test")
# wynik_wasSold_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/wynik_wasSold_test")

x_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/X_TEST_FULL").tocsr()[:elementsCount,:].tocoo()

klasyfikator_views = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/KLASYFIKATOR_VIEWS")
# klasyfikator_replies = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/ReportNinja2/KLASYFIKATOR_REPLIES")
# klasyfikator_wassold = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/ReportNinja2/KLASYFIKATOR_WASSOLD")


predicted = klasyfikator_views.predict(x_test)
roundedPredictedList = [ round(elem, 0) for elem in predicted ]
roundedPredicted = list(map(int, roundedPredictedList))
Y_test = wynik_views_test

poprawnosc = np.mean(abs(predicted - Y_test) < 3)

a = 9