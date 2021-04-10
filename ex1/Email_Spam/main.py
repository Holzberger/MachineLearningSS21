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

#%%
pickle_name    = "train_set_clean"
force_repickle = False
if path.exists(link_datafolder+pickle_name) and (not force_repickle):
    print("realoading cleaned dataset\n")
    [train_set_clean, test_set_clean] = pickle_data(link_folder=link_datafolder, 
                                                    name=pickle_name, operation="rb")
else:
    print("pickling cleaned dataset\n")
    train_set_clean = prep_mails(train_set['Body'])
    test_set_clean = prep_mails(test_set['Body'])
    pickle_data(data_item=[train_set_clean, test_set_clean], link_folder=link_datafolder, 
                name=pickle_name, operation="wb")
#%%
vectorizer_bow ,train_vec, test_vec =\
create_vectorizer(train_set_clean, test_set_clean, max_features=1000, max_count=0)
[train_vec, mask] = remove_outlyers(train_vec, threshold=100)
train_set.drop(train_set.index[mask], inplace=True)
#train_vec, test_vec = create_scaling(train_vec, test_vec)
#%%
knn = create_knn(train_vec, train_set['Label'], n_neighbors=5)
evaluate_algo(knn, test_vec, test_set['Label'], train_vec[:1000], train_set['Label'][:1000])
#%%
ppn = create_perceptron(train_vec, train_set['Label'], eta0=0.1, random_state=42,
                        max_iter=10,early_stopping=False)
evaluate_algo(ppn, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%
dtree = create_decisiontree(train_vec, train_set['Label'],
                            random_state=42, max_leaf_nodes=100)
evaluate_algo(dtree, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%
rndf = create_rnd_forrest(train_vec, train_set['Label'],
                            random_state=42, max_leaf_nodes=50)
evaluate_algo(rndf, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%
nb = create_nb(train_vec, train_set['Label'])
evaluate_algo(nb, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%

param_grid = {'eta0': [0.1,0.2,0.3,0.4],  
              'early_stopping': [True, False],
              'max_iter':[10,20,30]}  
ppn = create_perceptron(train_vec, train_set['Label'], grid=param_grid)
print(ppn.best_params_) 

#%%
param_grid = {'max_leaf_nodes':[5,10,20,30,40]}  
rndf = create_rnd_forrest(train_vec, train_set['Label'],grid=param_grid)
print(rndf.best_params_) 

