import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy

max_days = 1
data_dir = "D:\\Biblioteki\\Dokumenty (D)\\Studia\\ReportNinja2\\Ads"

index_query = 0
index_category = 1
index_count = 2


daily_queries = []
category_queries = []
colNames = ['KOLA', 'KOLB']
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
    # ,"photo_sizes"       # 18
    # ,"paidads_id_index"  # 19
    # ,"paidads_valid_to"  # 20
    #  "predict_sold"      # 21
     "predict_replies"   # 22
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

def convertString(txt):
    if txt is None:
        return ""
    txt = txt.strip('"')
    txt = txt.strip(',')
    txt = txt.strip('"')
    txt = txt.strip('\r')
    return txt

converters = {}
categories = []
grouped = []
usedCols = [22, 23]

p = os.path.join(data_dir, "testads.txt")

for file_index, file_name in enumerate(os.listdir(data_dir)):
        if file_index < max_days:
            p = os.path.join(data_dir, file_name)
            queries = pd.read_csv(p,  header=0, usecols=usedCols, names=col_names, converters=converters)
            # print(queries);

            # maxVal = queries["predict_views"].max()
            # plt.hist(queries["predict_views"], range(0, maxValue + step, step), log=True)

            queries[["predict_views", "predict_sold"]].groupby('predict_sold')['predict_views'].agg({'iloscOdtworzen': 'sum'}).plot()
            # queries["predict_views"].plot()
            # plt.axis([0, len(daily_queries), min(daily_queries) , max(daily_queries)])
            # plt.show()

            a = 9










