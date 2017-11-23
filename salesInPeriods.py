import pandas as pd
import os
import matplotlib.pyplot as plt


data_dir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Search_queries"
max_days = 14*8 #os.listdir(data_dir).__len__()

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
dni = dict()

# for i in range(1, 7+1):
#     dni[i] = 0
dayCounter = 1
monthIndex = 1
for file_index, file_name in enumerate(os.listdir(data_dir)):
    if file_index > (14*6)-1:
        if file_index < max_days:
            p = os.path.join(data_dir, file_name)

            queries = pd.read_csv(p,delimiter='","', engine="python", header =0, names = col_names, quotechar='"',
                                   converters=converters)


            # print(queries.loc[queries['category'] == '386']) # iphone
            if monthIndex not in dni:
                dni[monthIndex] = 0
            dni[monthIndex] += queries[['category','count']].loc[queries['category'] == '386'].groupby('category')['count'].agg({'counter':'sum'})['counter'].item()

    dayCounter += 1
    if (dayCounter > 14):
        dayCounter = 1
        monthIndex += 1


    print(str(file_index+1) + " / " + str(max_days) + " plik: " + file_name)



plt.bar(range(len(dni)), dni.values(), align='center')
plt.xticks(range(len(dni)), dni.keys())
plt.show()