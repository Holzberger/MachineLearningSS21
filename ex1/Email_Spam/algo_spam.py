from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def create_vectorizer(train_data, test_data, max_features=1, max_count=0):
    vectorizer = CountVectorizer(max_features=max_features)
    train_vectors = vectorizer.fit_transform(train_data).toarray()
    test_vectors = vectorizer.transform(test_data).toarray()
    if max_count>0:
        train_vectors[train_vectors>max_count] = max_count
        test_vectors[test_vectors>max_count] = max_count
    return vectorizer, train_vectors, test_vectors

def create_scaling(train_data, test_data):
    scaler = MinMaxScaler().fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)

def remove_outlyers(data, threshold=100):
    mask = (data>threshold).sum(axis=1)==0
    print("removing {} rows\n".format(np.sum(~mask)))
    return [data[mask], ~mask]

def create_knn(train_data, target, **kwargs):
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(train_data, target )
    return clf

def create_perceptron(train_data, target, grid=[], **kwargs):
    if grid != []:
        clf = GridSearchCV(Perceptron(), grid, refit = True, verbose = 3,n_jobs=2)
    else:
        clf = Perceptron(**kwargs)
    clf.fit(train_data, target)
    return clf

def create_decisiontree(train_data, target, **kwargs):
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(train_data, target)
    return clf

def create_rnd_forrest(train_data, target, grid=[], **kwargs):
    if grid != []:
        clf = GridSearchCV(RandomForestClassifier(), grid, refit = True, verbose = 3,n_jobs=2)
    else:
        clf = RandomForestClassifier(**kwargs)
    clf.fit(train_data, target)
    return clf

def create_nb(train_data, target):
    clf =MultinomialNB()
    clf.fit(train_data , target)
    return clf
    
def evaluate_algo(algo, test_data, test_target, train_data, train_target):
    print(algo)
    pred_test  = algo.predict(test_data)
    pred_train = algo.predict(train_data)
    print("testset report: \n",classification_report(test_target, pred_test))
    print("trainset report: \n",classification_report(train_target, pred_train))
    # print("accuracy of testset is {:.5} %".format(100*))
    # print("accuracy of trainset is {:.5} %".format(100*accuracy_score(train_target, pred_train)))
    # cmat = confusion_matrix(test_target, pred_test)
    # plt.figure(figsize = (6, 6))
    # sns.heatmap(cmat, annot = True, 
    #             cmap = 'Paired', 
    #             cbar = False, 
    #             fmt="d", 
    #             xticklabels=['Not Spam', 'Spam'], 
    #             yticklabels=['Not Spam', 'Spam']);
    

    
    
#=============================================================================








