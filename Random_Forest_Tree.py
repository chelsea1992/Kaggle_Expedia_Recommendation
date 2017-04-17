import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import ml_metrics as metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dtype={'is_booking':bool,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.int32,
        'user_location_country' : np.int32,
        'user_location_region' : np.int32,
        'user_location_city' : np.int32,
        'hotel_cluster' : np.int32,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}

df0 = pd.read_csv('train.csv',dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',')
df0['year'] = df0['date_time'].dt.year
train = df0.query('is_booking==True')

train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')

train['month'] = train['date_time'].dt.month
train['plan_time'] = ((train['srch_ci']-train['date_time'])/np.timedelta64(1,'D')).astype(float)
train['hotel_nights'] = ((train['srch_co']-train['srch_ci'])/np.timedelta64(1,'D')).astype(float)

m = train.orig_destination_distance.mean()
train['orig_destination_distance']=train.orig_destination_distance.fillna(m)
train.fillna(-1, inplace=True)

lst_drop=['date_time','srch_ci','srch_co']
train.drop(lst_drop,axis=1,inplace=True)

train2 = train.query('hotel_cluster==[2,6]')

print "Correlation"
print(train.corr())
pca = PCA(n_components=3,copy =True)
pca.fit(train2)
y = train2['hotel_cluster']
print "y",y
X = train2.drop('hotel_cluster',axis=1)
# pca.fit(train2)
print(pca.explained_variance_ratio_)
# print "X",X.head(2)
reduced = pca.transform(train2)
# print "reduced",reduced
#
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:,2],c=y,cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

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

y_train = train['hotel_cluster']
X_train = train.drop(['hotel_cluster','is_booking','year'],axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train,y_train, test_size=0.33)
#
rf_tree = RandomForestClassifier(n_estimators=31,max_depth=3,n_jobs=200,random_state=123)
rf_tree.fit(X_train2,y_train2)

importance = rf_tree.feature_importances_
indices=np.argsort(importance)[::-1][:10]
importance[indices]

plt.barh(range(5), importance[indices],color='r')
plt.yticks(range(5),X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.show()

# y_pred=rf_tree.predict_proba(test)
y_pred=rf_tree.predict_proba(X_test2)
#take largest 5 probablities' indexes
a=y_pred.argsort(axis=1)[:,-5:]

dict_cluster = {}
for (k,v) in enumerate(rf_tree.classes_):
    dict_cluster[k] = v

b = []
for i in a.flatten():
    b.append(dict_cluster.get(i))
predict_class = np.array(b).reshape(a.shape)
predict_class = map(lambda x: ' '.join(map(str,x)), predict_class)

num=0
with open("rf_pred.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    for row in predict_class:
        outfile.write("%d,%s\n" % (num, row))
        num += 1

