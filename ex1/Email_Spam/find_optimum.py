from extractor import *

from algo_spam import *

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import os.path
from os import path

#%%

#link_datafolder = "C:\\Users\\saeny\\Desktop\\machine learning\\checkdatasets\\"
link_datafolder = "./../../../checkdatasets/"

dataset = import_email_data(link_folder=link_datafolder)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

train_set = remove_duplicates(train_set, remove_dups=True, print_dups=False)
train_set = remove_missing_vals(train_set, remove_missing=True, print_missing=False)

test_set1  = import_dataset(link_folder=link_datafolder, name_dataset="completeSpamAssassin.csv")
test_set1 = remove_duplicates(test_set1, remove_dups=True, print_dups=True)
test_set1 = remove_missing_vals(test_set1, remove_missing=True, print_missing=True)

tmp = test_set1
test_set1 = test_set
test_set=tmp

#%%
pickle_name    = "datasets_optimum_clean"
force_repickle = True
if path.exists(link_datafolder+pickle_name) and (not force_repickle):
    print("realoading cleaned dataset\n")
    [train_set_clean, 
     test_set_clean, 
     test_set_clean1] = pickle_data(link_folder=link_datafolder, 
                                    name=pickle_name, operation="rb")
else:
    print("pickling cleaned dataset\n")
    train_set_clean = prep_mails(train_set['Body'])
    test_set_clean = prep_mails(test_set['Body'])
    test_set_clean1 = prep_mails(test_set1['Body'])
    pickle_data(data_item=[train_set_clean, 
                           test_set_clean,
                           test_set_clean1], link_folder=link_datafolder, 
                name=pickle_name, operation="wb")
#%%
vectorizer_bow ,train_vec, test_vec, test_vec1 =\
create_vectorizer(train_set_clean, test_set_clean, test_set_clean1, max_features=50)
[train_vec, train_set] = remove_outlyers(train_vec, train_set, threshold=100)
#%%
ppn = create_perceptron(train_vec, train_set['Label'], eta0=0.3, random_state=42,
                        max_iter=100,early_stopping=False)
evaluate_algo(ppn, test_vec1, test_set1['Label'], train_vec, train_set['Label'])
#%%
rndf = create_rnd_forrest(train_vec, train_set['Label'],
                            random_state=42, max_leaf_nodes=50)
evaluate_algo(rndf, test_vec1, test_set1['Label'], train_vec, train_set['Label'])
#%%
nb = create_nb(train_vec, train_set['Label'])
evaluate_algo(nb, test_vec1, test_set1['Label'], train_vec, train_set['Label'])


