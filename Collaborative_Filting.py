from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import ml_metrics as metrics

# train data
dtype={'is_booking':bool,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'hotel_cluster' : np.int32,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}

df0 = pd.read_csv('train.csv',dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',')
df0['year'] = df0['date_time'].dt.year
train = df0.query('is_booking==True')
print train.size

train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')
#
train['month'] = train['date_time'].dt.month
train['plan_time'] = ((train['srch_ci']-train['date_time'])/np.timedelta64(1,'D')).astype(float)
train['hotel_nights'] = ((train['srch_co']-train['srch_ci'])/np.timedelta64(1,'D')).astype(float)
m = train.orig_destination_distance.mean()
train['orig_destination_distance']=train.orig_destination_distance.fillna(m)
train.fillna(-1, inplace=True)
#
lst_drop=['date_time','srch_ci','srch_co']
train.drop(lst_drop,axis=1,inplace=True)


#test data
dtype1={'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt': np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country': np.str_,
        'user_location_region': np.str_,
        'user_location_city' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}
test = pd.read_csv('test.csv',dtype=dtype1,usecols=dtype1,parse_dates=['date_time'] ,sep=',',nrows = 1000)
test['srch_ci']=pd.to_datetime(test['srch_ci'],infer_datetime_format = True,errors='coerce')
test['srch_co']=pd.to_datetime(test['srch_co'],infer_datetime_format = True,errors='coerce')

test['month']=test['date_time'].dt.month
test['plan_time'] = ((test['srch_ci']-test['date_time'])/np.timedelta64(1,'D')).astype(float)
test['hotel_nights']=((test['srch_co']-test['srch_ci'])/np.timedelta64(1,'D')).astype(float)

n=test.orig_destination_distance.mean()
test['orig_destination_distance']=test.orig_destination_distance.fillna(m)
test.fillna(-1,inplace=True)
lst_drop=['date_time','srch_ci','srch_co']
test.drop(lst_drop,axis=1, inplace=True)

print "test",test.shape

most_frequent = {}

for k in destination_clusters:
    # print "destination_clusters[k]",destination_clusters[k]
    most_frequent[k] = get_top5(destination_clusters[k])
    
match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']
groups = train.groupby(match_cols)
preds = []

for i, row in test.iterrows():
        # print row["srch_destination_id"]
        preds.append(most_frequent[row["srch_destination_id"]])


def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(test.shape[0]):
    exact_matches.append(generate_exact_matches(test.iloc[i], match_cols))

def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

for p in range(len(preds)):
        print "exact_matches[p]", exact_matches[p]

full_preds = [f5(exact_matches[p] + preds[p])[:5] for p in range(len(preds))]
write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(test["id"][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_clusters"] + write_frame
print "test_1"
with open("predictions2.csv", "w+") as f:
    f.write("\n".join(write_frame))
