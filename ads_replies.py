import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np

max_days = 1
data_dir = "data\\Ads"

index_query = 0
index_category = 1
index_count = 2


daily_queries = []
category_queries = []
col_names = [
     "id"                 # 0
    # ,"region_id"         # 1
     ,"category_id"       # 2
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
    #,"predict_sold"      # 21
    ,"predict_replies"   # 22
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
    return txt

converters = {21:convertSold}
categories = []
grouped = []
mean_sold = [];
max_sold = [];
std_sold = [];
mean_viewed = [];
max_viewed = [];
std_viewed = [];
monthly_sold = [];
viewed_limits = dict();

usedCols = [0, 2 ,22]
months = [];
files = os.listdir(data_dir)
print(len(files))
p = os.path.join(data_dir, "testads.txt")
start = time.time()
for file_index, file_name in enumerate(os.listdir(data_dir)):
        if file_index < len(files):
            p = os.path.join(data_dir, file_name)
            queries = pd.read_csv(p,  header=0, usecols=usedCols, names=col_names, converters=converters)
            # months.append(file_name)
            viewed =  queries[["id", "predict_replies"]].groupby('id')['predict_replies'].agg({'viewed': 'sum'});
            nonzeros = viewed.loc[viewed['viewed'] != 0]
            #print (nonzeros.loc[nonzeros['viewed']  < 2])
            #mean_viewed.append(sorted_viewed['viewed'].mean())
            # max = max(viewed['viewed']);
            # max_viewed.append(max)
            # #std_viewed.append(sorted_viewed['viewed'].std())
            for key in range(2,100):
                drop_df = nonzeros.loc[nonzeros['viewed']  < key]
                drop = len(drop_df.index)
                if (key) not in viewed_limits.keys():
                    viewed_limits[key] = drop
                else:
                    viewed_limits[key] += drop #pd.concat([viewed_limits[key*1000],drop])
                nonzeros = pd.concat([nonzeros,drop_df]).drop_duplicates(keep=False)
            print(str(file_index + 1) + " / " + str(len(files)) + " plik: " + file_name)
            # sold = queries[["category_id", "predict_sold"]].groupby('category_id')['predict_sold'].agg({'sold': 'sum'});
            # sorted_sold = sold.sort_values(by=['sold'], ascending= False)
            # mean_sold.append(va['sold'].mean())
            # max_sold.append(max(va['sold']))
            # std_sold.append(va['sold'].std())

            # sold_monthly = queries['predict_sold'].sum();
            # monthly_sold.append(sold_monthly);

end = time.time()
#print(viewed)
plt.bar(range(len(viewed_limits)), viewed_limits.values(), align='center')
plt.xticks(range(len(viewed_limits)), viewed_limits.keys())
plt.ylim(0,50)
plt.show()
print(viewed_limits)
print (end-start);
# print (mean_viewed);
# print (max_viewed);
# print (std_viewed);
# data =  np.squeeze(monthly_sold);
# print(data)
#
# y_pos = np.arange(len(monthly_sold))
# x = len(monthly_sold)
# plt.bar(y_pos,monthly_sold,align='center',alpha=0.5)
# plt.xticks(y_pos, months)
# plt.ylabel('Sold number')
# plt.show();
# print (mean_sold);
# print (max_sold);
# print (std_sold);








