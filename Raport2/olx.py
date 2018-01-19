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



# #CECHY TEKSTOWE ========================================================================
# #Pobranie plików z cechami
# cecha_descriptions = loadAttributeFileAsLines('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/titles/titles0')
# cecha_desc_train, cecha_dest_test = splitLines(cecha_descriptions)
#
# #Zamiana słów na sechy
# count_vect_titles = CountVectorizer(stop_words=['a', 'i', 'o', 'na', 'w', 'po'], max_features = 1000)
# X_train_counts = count_vect_titles.fit_transform(cecha_desc_train) #fit: robi listę słów i przydziela im indeksy#transform - zlicza listę słów
# X_test_counts = count_vect_titles.transform(cecha_dest_test)
#
# #Tfidf na cechach tekstowych
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) #dopasowanie do danych i przekształcenie macierzy
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# saver.dump(count_vect_titles, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/CV_titles")
# saver.dump(tfidf_transformer, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TFIDF_titles")
#
#
#
# #=====================================================================================
#
# cecha_descriptions = loadAttributeFileAsLines('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/descriptions/descriptions0')
# cecha_desc_train, cecha_dest_test = splitLines(cecha_descriptions)
#
# #Zamiana słów na sechy
# count_vect_descriptions = CountVectorizer(stop_words=['a', 'i', 'o', 'na', 'w', 'po'], max_features = 1000)
# X_train_counts = count_vect_descriptions.fit_transform(cecha_desc_train) #fit: robi listę słów i przydziela im indeksy#transform - zlicza listę słów
# X_test_counts = count_vect_descriptions.transform(cecha_dest_test)
#
# #Tfidf na cechach tekstowych
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) #dopasowanie do danych i przekształcenie macierzy
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# saver.dump(count_vect_descriptions, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/CV_descriptions")
# saver.dump(tfidf_transformer, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TFIDF_descriptions")


#WYNIK : wasSold
# lines_wasSold = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/wasSold/WasSold0")
# wynik_wasSold_train, wynik_wasSold_test = splitLines(lines_wasSold)
# wynik_wasSold_train = list(map(int, wynik_wasSold_train))
# wynik_wasSold_test = list(map(int, wynik_wasSold_test))
#
# #WYNIK : replies
# lines_replies = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies")
# wynik_replies_train, wynik_replies_test = splitLines(lines_replies)
# wynik_replies_train = list(map(int, wynik_replies_train))
# wynik_replies_test = list(map(int, wynik_replies_test))
#
# #WYNIK : views
# lines_views = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_views")
# wynik_views_train, wynik_views_test = splitLines(lines_views)
# wynik_views_train = list(map(int, wynik_views_train))
# wynik_views_test = list(map(int, wynik_views_test))



#CECHA : hasPicture
#hasPicture_train = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/pictures/hasPicture0")
# hasPicture_OLX = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXpictureCount")

#CECHA : age
# age_train = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_age")
# age_OLX = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXage")

#CECHA : hasNumber_description
# hasNumber_description_train = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_hasNumber_description")
hasNumber_description_OLX = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXage")

#CECHA : hasNumber_title
# hasNumber_title_train = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/cecha_hasNumber_title")
hasNumber_title_OLX = loadAttributeFromFile("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXcontainsNumber")

#CV i TFIDF
# cv_titles = loadObject("D:\Biblioteki\Dokumenty (D)\Studia\ReportNinja2\CV_titles")
# cv_descriptions = loadObject("D:\Biblioteki\Dokumenty (D)\Studia\ReportNinja2\CV_descriptions")
#
# tfidf_titles = loadObject("D:\Biblioteki\Dokumenty (D)\Studia\ReportNinja2\TFIDF_titles")
# tfidf_descriptions = loadObject("D:\Biblioteki\Dokumenty (D)\Studia\ReportNinja2\TFIDF_descriptions")

#OLX titles
# olx_titles = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXtitles")
#OLD descriptions
# olx_descriptions = loadAttributeFileAsLines("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLXdescriptions")

#Zamiana słów na sechy
# olx_titles_CV  = cv_titles.transform(olx_titles)
# olx_descriptions_CV  = cv_descriptions.transform(olx_descriptions)

#Tfidf na cechach tekstowych
# olx_titles_tfidf = tfidf_titles.transform(olx_titles_CV)
# olx_descriptions_tfidf = tfidf_descriptions.transform(olx_descriptions_CV)

# olx_titles_tfidf =  loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_TITLES_TFIDF")
# olx_descriptions_tfidf = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_DESCRIPTIONS_TFIDF")

# olx_cechy_tekstowe = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_CECHY_TEKSTOWE")
#
# X_test = olx_cechy_tekstowe
# X_test = dodajCeche(X_test, hasPicture_OLX)
# X_test = dodajCeche(X_test, age_OLX)

# X_test = loadObject("D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_CECHY_12")
# X_test = dodajCeche(X_test, hasNumber_title_OLX)
# X_test = dodajCeche(X_test, hasNumber_description_OLX)

saveObject(X_test, "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/OLX_CECHY")