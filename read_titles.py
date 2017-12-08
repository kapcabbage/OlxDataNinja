import pandas as pd
import os
import time
import pl_stemmer
import matplotlib.pyplot as plt
import numpy

max_days = 1
data_dir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Ads"

index_query = 0
index_category = 1
index_count = 2


daily_queries = []
category_queries = []
col_names = [
    # "id"                 # 0
    # ,"region_id"         # 1
     #"category_id"       # 2
    # ,"subregion_id"      # 3
    # ,"district_id"       # 4
    # ,"city_id"           # 5
    # ,"accurate_location" # 6
    # ,"user_id"           # 7
    # ,"sorting_date"      # 8
    # ,"created_at_first"  # 9
    # ,"valid_to"          # 10
    "title"             # 11
     #,"description"       # 12
    # ,"full_description"  # 13
    # ,"has_phone"         # 14
    # ,"params"            # 15
    # ,"private_business"  # 16
    # ,"has_person"        # 17
    # ,"photo_sizes"       # 18
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

def convertString(txt):
    if txt is None:
        return ""
    txt = txt.replace('\n','')
    txt = txt.replace('\t','')
    txt = txt.replace('\n\n','')
    txt = txt.replace('\r', '')

    #stemowanie (usuwanie polskich odmian)
    result = ""
    for word in txt.split(" "):

        stem = pl_stemmer.remove_nouns(word)
        stem = pl_stemmer.remove_diminutive(stem)
        stem = pl_stemmer.remove_adjective_ends(stem)
        stem = pl_stemmer.remove_verbs_ends(stem)
        stem = pl_stemmer.remove_adverbs_ends(stem)
        stem = pl_stemmer.remove_plural_forms(stem)
        stem = pl_stemmer.remove_general_ends(stem)
        result += " " + stem;

    return result

converters = {11: convertString}
categories = []
grouped = []
mean_sold = [];
max_sold = [];
std_sold = [];
usedCols = [11]


start = time.time()

p = os.path.join(data_dir, "001_anonimized_9")

queries = pd.read_csv(p,  header=0, usecols=usedCols, names=col_names, converters=converters)

print(queries)

outputDir = "D:/Biblioteki/Dokumenty (D)/Studia/ReportNinja2/Cechy/titles"
queries.to_csv(outputDir, index=False);
pl_stemmer



end = time.time()
print (end-start);
print (mean_sold);
print (max_sold);
print (std_sold);









