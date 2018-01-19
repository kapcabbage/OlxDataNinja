#PRZYKLAD

# categories = ['misc.forsale', 'rec.autos',
#               'rec.motorcycles', 'sci.space',
#                'sci.crypt']
#
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.externals import joblib as saver
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#
# print(twenty_train.target_names)  # w target_names mamy dostęp do listy nazw kategorii
# print(len(twenty_train.data))  # w data mamy dostęp do danych
# print(len(twenty_train.filenames))  # filenames zwróci nam listę nazw plików

#Przykład testowy dodawania cechy

from sklearn.externals import joblib as saver
import numpy as np
import scipy.sparse as sparse

#Pobranie plików z cechami
#text_file = open('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/titles/titles0.txt', "r")
filename = 'D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Testy/przykład1.txt'
filename2 = 'D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Testy/przykładCechy.txt'
#text_file = open('D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Testy/przykład1.txt', 'r')
#lines = text_file.readlines()
with open(filename) as f:
    lines = f.read().splitlines()
c_titles = lines

with open(filename2) as f:
    lines2 = f.read().splitlines()
c_titles2 = lines2

#Zamiana słów na sechy
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words=['a', 'i', 'o', 'na', 'w', 'po'], max_features = 1000)
X_train_counts = count_vect.fit_transform(c_titles)
#fit: robi listę słów i przydziela im indeksy
#transform - zlicza listę słów

#dodanie cechy manualnej


import numpy as np
from scipy.sparse import csr_matrix

def linesToCSRMatrix(lines):
    row = np.arange(len(lines))
    col = np.zeros(len(lines))
    data = np.array(lines).astype(float)

    return csr_matrix((data, (row, col)))


c2 = linesToCSRMatrix(c_titles2)


# row = np.arange(len(c_titles2))
# col = np.zeros(len(c_titles2))
# data = np.array(c_titles2).astype(float)
# cecha2 = csr_matrix((data, (row, col))).toarray()



X_cechy = sparse.hstack((X_train_counts, c2))

a = 9

#dodanie cechy manualnej
#X_cechy = sparse.hstack((X_train_counts,np.array(lines2).astype(float)[:,None])).tocsr()