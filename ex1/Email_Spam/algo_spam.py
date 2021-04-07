from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

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

def create_knn(train_data, target, n_neighbors=1):
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(train_data, target )
    return clf

def create_perceptron(train_data, target, eta=0.1, n_iter=50, random_state=42):
    clf = Perceptron(random_state=random_state, 
                     eta0=eta)
    clf.fit(train_data, target)
    return clf

def create_decisiontree(train_data, target, random_state=42, max_leaf_nodes=10):
    clf = DecisionTreeClassifier(random_state=0,max_leaf_nodes=max_leaf_nodes)
    clf.fit(train_data, target)
    return clf

def create_rnd_forrest(train_data, target, random_state=42, max_leaf_nodes=10):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train_data, target)
    return clf

def create_nb(train_data, target):
    clf =MultinomialNB()
    clf.fit(train_data , target)
    return clf
    
def evaluate_algo(algo, test_data, test_target, train_data, train_target):
    print(algo)
    pred_test  = algo.predict(test_data)
    print("accuracy of testset is {:.5} %".format(100*accuracy_score(test_target, pred_test)))
    pred_train = algo.predict(train_data)
    print("accuracy of trainset is {:.5} %".format(100*accuracy_score(train_target, pred_train)))
    cmat = confusion_matrix(test_target, pred_test)
    plt.figure(figsize = (6, 6))
    sns.heatmap(cmat, annot = True, 
                cmap = 'Paired', 
                cbar = False, 
                fmt="d", 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam']);