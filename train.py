#!/usr/bin/env python3

import datetime
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.externals import joblib

df = pd.read_json('./data/news.json')
df = df[['body', 'subject', 'language', 'categories']]
df.columns = ['body', 'subject', 'language', 'categories']
df = df[pd.notnull(df['body'])]
df = df.loc[(df['language'] == 'English')]
df['categories']   = df['categories'].apply(lambda x: x.title())

df['cat_id']       = df['categories'].factorize()[0]
df['lang_id']      = df['language'].factorize()[0]
df['char_count']   = df['body'].apply(len)
df['word_count']   = df['body'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)

df = df[df.categories != 'Frontpage']
df = df[df.categories != 'Uncategorized']
df = df.loc[(df.groupby('categories').cumcount() > 8)]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', ngram_range=(1, 2), max_features=5000)
features = tfidf.fit_transform(df.body).toarray()
labels = df.cat_id

xtrain, xtest, ytrain, ytest = train_test_split(df['body'], df['categories'], test_size=0.33, random_state=42)

engines = [
    ('RFC', RandomForestClassifier()),
    ('GBC', GradientBoostingClassifier()),
    ('PAC', PassiveAggressiveClassifier()),
    ('RC', RidgeClassifier()),
    ('RCCV', RidgeClassifierCV()),
    ('ETC', ExtraTreesClassifier()),
    ('KNC', KNeighborsClassifier(n_neighbors=10)),
    ('MNB', MultinomialNB())
]
engines_dt = []
today = datetime.date.today()
for name, engine in engines:
    model = make_pipeline(tfidf, engine)
    model.fit(xtrain, ytrain)
    prediction  = model.predict(xtest)
    score = model.score(xtest, prediction)
    # print("{}: Score: {} Accuracy: {:.2f}".format(name, score, accuracy_score(ytest, prediction)))
    engines_dt.append([name, score, accuracy_score(ytest, prediction), today])
    
with open('./data/lsvc.pickle', 'wb') as f:
    joblib.dump(model, f)
    
df_dt = pd.DataFrame(engines_dt, columns=['engine', 'score', 'accuracy', 'date'])

print(df_dt)