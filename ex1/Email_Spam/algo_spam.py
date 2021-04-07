from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.linear_model import Perceptron


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
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(train_data, target )
    return knn

def create_perceptron(train_data, target, eta=0.1, n_iter=50, random_state=42):
    ppn = Perceptron(random_state=random_state, 
                     eta0=eta)
    ppn.fit(train_data, target)
    return ppn
    
def evaluate_algo(algo, test_data, target):
    pred_test = algo.predict(test_data)
    print("accuracy is {:.5} %".format(100*accuracy_score(target, pred_test)))
    cmat = confusion_matrix(target, pred_test)
    plt.figure(figsize = (6, 6))
    sns.heatmap(cmat, annot = True, 
                cmap = 'Paired', 
                cbar = False, 
                fmt="d", 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam']);