import pandas as pd
import os
import matplotlib.pyplot as plt


data_dir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Search_queries"
max_days = 49

index_query = 0
index_category = 1
index_count = 2

daily_queries = []
category_queries = []
col_names = ["query", "category", "count"]

def convert_query(a):
    if a is None:
        return ""
    return str(a.strip('"'))

def convert_category(a):
    if a is None:
        return -1
    a = a.strip('"')
    a = a.strip(',')
    a = a.strip('"')
    return a
    #return int(a)

def convert_count(a):
    if a is None:
        return -1
    a = a.strip('"')
    a = a.strip(',')
    a = a.strip('"')
    return int(a)

converters = {index_query:convert_query, index_category:convert_category, index_count: convert_count}
categories = []
grouped = []
sum = 0
naned = 0
dniTygodnia = dict()

# for i in range(0,14):
#     dniTygodnia[i] = 2*i



for i in range(1, 7+1):
    dniTygodnia[i] = 0

for file_index, file_name in enumerate(os.listdir(data_dir)):
        if file_index < max_days:
            p = os.path.join(data_dir, file_name)

            queries = pd.read_csv(p,delimiter='","', engine="python", header =0, names = col_names, quotechar='"',
                                   converters=converters)

            dniTygodnia[(file_index%7)+1] += queries['count'].agg({'counter':'sum'}).item();

            print(str(file_index+1) + " / " + str(max_days))



plt.bar(range(len(dniTygodnia)), dniTygodnia.values(), align='center')
plt.xticks(range(len(dniTygodnia)), dniTygodnia.keys())
plt.show()