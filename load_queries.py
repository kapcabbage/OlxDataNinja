import pandas as pd
import os
import matplotlib.pyplot as plt

max_days = 1
data_dir = "H:\\Pobrane\\DataNinja\\search_queries_2016_11_01"

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
for file_index, file_name in enumerate(os.listdir(data_dir)):
        if file_index < max_days:
            p = os.path.join(data_dir, file_name)
            queries = pd.read_csv(p,delimiter='","', engine="python", header =0, names = col_names, quotechar='"',
                                   converters=converters)

            #naned = queries['category'].isnull().sum();
            queries['category'].fillna(0,inplace=True)
            #nani = queries.loc[queries['category'] == "-1"]
            print queries
            #print queries
            grouped = queries[['category','count']].groupby('category')['count'].agg({'counter':'sum'}).reset_index()
            print grouped

            naned = grouped['counter'].sum()
            #for key, item in grouped:
             #   new = grouped.get_group(key)
              #  group = new[['count']].agg('sum')
               # print group
                #sum+= grouped.get_group(key)['count'].sum();
                #category_queries
            daily_queries.append(queries['count'].sum())
            grouped.plot()
            # plt.axis([0, len(daily_queries), min(daily_queries) , max(daily_queries)])
            plt.show()
print daily_queries[0]
print sum + naned
