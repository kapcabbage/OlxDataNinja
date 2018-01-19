import pandas as pd
import os
import time
import json
import pl_stemmer
import re
import matplotlib.pyplot as plt
import numpy

max_days = 1
data_dir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXINPUT/"
outputDir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/OLXOUTPUT/"


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
     "photo_sizes"       # 18
    # ,"paidads_id_index"  # 19
    # ,"paidads_valid_to"  # 20
    #,"predict_sold"      # 21
    #,"predict_replies"   # 22
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

def convertSold(number):
    #print(number)
    if number == None:
        return 0
    if number == 'f':
        return 0
    if number == 't':
        return 1

def convertPhotoSizes(txt):
    if txt.strip() is None:
        return ""
    else:
        pictureCount = 0
        try:
            jsonVar = json.loads(txt)
            if jsonVar != None:
                pictureCount = len(jsonVar)
        except ValueError:
            pictureCount = 0

        return pictureCount

converters = {18: convertPhotoSizes}
categories = []
grouped = []
mean_sold = [];
max_sold = [];
std_sold = [];
usedCols = [18]


start = time.time()

print ("Rozpoczęto!...");

iterator = 0
for file_index, file_name in enumerate(os.listdir(data_dir)):
    if iterator > 1:
        break
    inputFile = data_dir + "/" + file_name
    outputFile = outputDir + str(iterator)
    iterator += 1
    queries = pd.read_csv(inputFile, usecols=usedCols, names=col_names, converters=converters)

    data1 = queries.iloc[1:len(queries) - 1] #Bez nagłówka (1 linijki)
    data2 = queries.iloc[[len(queries) - 1]]
    data1.to_csv(outputFile, sep=',', encoding='utf-8', index=False, header=False)
    data2.to_csv(outputFile, sep=',', encoding='utf-8', index=False, header=False,  mode='a', line_terminator="")





end = time.time()
print (end-start);
print (mean_sold);
print (max_sold);
print (std_sold);









