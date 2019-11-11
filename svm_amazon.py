import pandas as pd
import numpy as np

from sklearn.svm import SVR

train = pd.read_csv('train.csv')


test = pd.read_csv('test.csv')

print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))

train.head(5)
test.head(5)

train.apply(lambda x: len(x.unique()))

import itertools
target = "ACTION"
col4train = [x for x in train.columns if x!=target]

col1 = 'ROLE_CODE'
col2 = 'ROLE_TITLE'

pair = len(train.groupby([col1,col2]).size())
single = len(train.groupby([col1]).size())

print(col1, col2, pair, single)

col4train = [x for x in col4train if x!='ROLE_TITLE']

#linear - OHE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')

X = ohe.fit_transform(train[col4train])
y = train["ACTION"].values

from sklearn.model_selection import cross_validate
from catboost import CatBoostRegressor

best_params = {'bagging_temperature': 0.6,
               'border_count': 200,
               'depth': 8,
               'iterations': 350,
               'l2_leaf_reg': 30,
               'learning_rate': 0.30,
               'random_strength': 0.01,
               'scale_pos_weight': 0.48}

#model = CatBoostRegressor(iterations=1000, depth=3, learning_rate=0.1, loss_function='RMSE')

model = SVR(kernel = "rbf")





stats = cross_validate(model, X, y, groups=None, scoring='roc_auc',
                       cv=5, n_jobs=2, return_train_score = True)
stats = pd.DataFrame(stats)
stats.describe().transpose()



X = ohe.fit_transform(train[col4train])
y = train["ACTION"].values
X_te = ohe.transform(test[col4train])

model.fit(X,y)
predictions = model.predict(X_te)

submit = pd.DataFrame()
submit["Id"] = test["id"]
submit["ACTION"] = predictions

submit.to_csv("svm_predicted.csv", index = False)

