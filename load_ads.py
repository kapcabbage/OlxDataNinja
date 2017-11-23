import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy

max_days = 1
data_dir = "data\\Ads"

index_query = 0
index_category = 1
index_count = 2


daily_queries = []
category_queries = []
col_names = [
    # "id"                 # 0
    # ,"region_id"         # 1
     "category_id"       # 2
    # ,"subregion_id"      # 3
    # ,"district_id"       # 4
    # ,"city_id"           # 5
    # ,"accurate_location" # 6
    # ,"user_id"           # 7
    # ,"sorting_date"      # 8
    # ,"created_at_first"  # 9
    # ,"valid_to"          # 10
    # ,"title"             # 11
     #,"description"       # 12
    # ,"full_description"  # 13
    # ,"has_phone"         # 14
    # ,"params"            # 15
    # ,"private_business"  # 16
    # ,"has_person"        # 17
    # ,"photo_sizes"       # 18
    # ,"paidads_id_index"  # 19
    # ,"paidads_valid_to"  # 20
    ,"predict_sold"      # 21
    ,"predict_replies"   # 22
    ,"predict_views"     # 23
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
    return txt

converters = {21:convertSold}
categories = []
grouped = []
mean_sold = [];
max_sold = [];
std_sold = [];
usedCols = [2, 21,22, 23]
files = os.walk(data_dir).__next__()
print(len(files))
p = os.path.join(data_dir, "testads.txt")
start = time.time()
for file_index, file_name in enumerate(os.listdir(data_dir)):
        if file_index < len(files):
            p = os.path.join(data_dir, file_name)

            queries = pd.read_csv(p,  header=0, usecols=usedCols, names=col_names, converters=converters)

            sa = queries[["category_id", "predict_sold"]].groupby('category_id')['predict_sold'].agg({'sold': 'sum'});
            va = sa.sort_values(by=['sold'], ascending= False)
            mean_sold.append(va['sold'].mean())
            max_sold.append(max(va['sold']))
            std_sold.append(va['sold'].std())

end = time.time()
print (end-start);
print (mean_sold);
print (max_sold);
print (std_sold);









