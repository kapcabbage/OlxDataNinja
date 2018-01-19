import pandas as pd
import os
import time
import json
import pl_stemmer
import re
import matplotlib.pyplot as plt
import numpy

max_days = 1
data_dir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Ads"
outputDir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/TworzenieRaportu/wynik_replies"

index_query = 0
index_category = 1
index_count = 2


daily_queries = []
category_queries = []
col_names = [
    # "id"                 # 0
    # ,"region_id"         # 1
    # ,"category_id"       # 2
    # ,"subregion_id"      # 3
    # ,"district_id"       # 4
    # ,"city_id"           # 5
    # ,"accurate_location" # 6
    # ,"user_id"           # 7
    # ,"sorting_date"      # 8
    # ,"created_at_first"  # 9
    # ,"valid_to"          # 10
    # ,"title"             # 11
    # ,"description"       # 12
    # ,"full_description"  # 13
    # ,"has_phone"         # 14
    # ,"params"            # 15
    # ,"private_business"  # 16
    # ,"has_person"        # 17
    # "photo_sizes"       # 18
    # ,"paidads_id_index"  # 19
    # ,"paidads_valid_to"  # 20
    # ,"predict_sold"      # 21
    "predict_replies"   # 22
    #,"predict_views"     # 23
    # ,"reply_call"        # 24
    # ,"reply_sms"         # 25
    # ,"reply_chat"        # 26
    # ,"reply_call_intent" # 27
    # ,"reply_chat_intent" # 28
]

def convertID(txt):
    if txt is None:
        return ""
    return str(txt.strip('"'))

def convertReplies(number):
    #print(number)
    return int(number)


converters = {22: convertReplies}
categories = []
grouped = []
mean_sold = [];
max_sold = [];
std_sold = [];
usedCols = [22]


start = time.time()

print ("Rozpoczęto!...");

iterator = 0
for file_index, file_name in enumerate(os.listdir(data_dir)):
    if ( iterator > 0):
        break

    inputFile = data_dir + "/" + file_name
    outputFile = outputDir # + str(iterator)
    iterator += 1
    queries = pd.read_csv(inputFile,  header=0, usecols=usedCols, names=col_names, converters=converters)

    data1 = queries.iloc[0:len(queries) - 1]
    data2 = queries.iloc[[len(queries) - 1]]
    data1.to_csv(outputFile, sep=',', encoding='utf-8', header=False, index=False)
    data2.to_csv(outputFile, sep=',', encoding='utf-8', header=False, index=False, mode='a', line_terminator="")

    #queries.to_csv(outputFile, index=False, header=None);





end = time.time()
print (end-start);
print (mean_sold);
print (max_sold);
print (std_sold);









