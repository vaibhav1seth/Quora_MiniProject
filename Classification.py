# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
# import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/test.csv")
train_df.head()

train_df['target'].value_counts()

from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

train_text = train_df['question_text']
test_text = test_df['question_text']
train_target = train_df['target']
all_text = train_text.append(test_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_text)

count_vectorizer = CountVectorizer(max_df=0.8,ngram_range=(1, 1))
count_vectorizer.fit(all_text)

kfold = KFold(n_splits = 5, shuffle = True, random_state = 43)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
test_preds = 0
oof_preds = np.zeros([train_df.shape[0],])

train_text_features_cv = count_vectorizer.transform(train_text)
test_text_features_cv = count_vectorizer.transform(test_text)

# train_text_features_tf = tfidf_vectorizer.transform(train_text)
# test_text_features_tf = tfidf_vectorizer.transform(test_text)

from sklearn import svm
test_preds = 0
# for i, (train_idx,valid_idx) in enumerate(kfold.split(train_df)):
for i ,(train_idx,valid_idx) in enumerate(skf.split(train_df, train_df['target'])):
    x_train, x_valid = train_text_features_cv[train_idx,:], train_text_features_cv[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier = LogisticRegression(C=3,solver='lbfgs',max_iter=5000,penalty='l2')
    # liblinear can not use more than one processors
#     classifier = RandomForestClassifier(n_estimators=100, max_depth=5,n_jobs=-1)
    
#     classifier = svm.SVC()
    print('fitting.......')
    classifier.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier.predict_proba(test_text_features_cv)[:,1]


print("done modelling !!!")    
pred_train = (oof_preds > .15).astype(np.int)
print(pred_train)
f1 =f1_score(train_target, pred_train)
print(f1)
# rf_df= pd.DataFrame()
# np.array(oof_preds).max()

submission1 = pd.DataFrame.from_dict({'qid': test_df['qid']})
submission1['prediction'] = (test_preds>0.14).astype(np.int)
submission1.to_csv('submission.csv', index=False)
submission1['prediction'] = (test_preds>0.14)

#print(test_df)
print(submission1.shape)
print(submission1['prediction'])

print("project done !!!!")
