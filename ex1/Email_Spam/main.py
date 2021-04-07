from extractor import import_email_data,\
                      remove_duplicates,\
                      remove_missing_vals,\
                      prep_mails
                      
from algo_spam import *
                      
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


#%%

link_datafolder = "../../../checkdatasets/"

dataset = import_email_data(link_folder=link_datafolder)
#%%

train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)


remove_duplicates(train_set, remove_dups=True, print_dups=False)
remove_missing_vals(train_set, remove_missing=True, print_missing=False)


#%%
train_set_clean = prep_mails(train_set['Body'])
test_set_clean = prep_mails(test_set['Body'])
#%%
vectorizer_bow ,train_vec, test_vec =\
create_vectorizer(train_set_clean, test_set_clean, max_features=3000, max_count=10)
#%%
knn = create_knn(train_vec, train_set['Label'], n_neighbors=5)
evaluate_algo(knn, test_vec, test_set['Label'], train_vec[:1000], train_set['Label'][:1000])
#%%
ppn = create_perceptron(train_vec, train_set['Label'], eta=0.1, 
                        n_iter=50, random_state=42)
evaluate_algo(ppn, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%
dtree = create_decisiontree(train_vec, train_set['Label'], 
                            random_state=42, max_leaf_nodes=100)
evaluate_algo(dtree, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%%
rndf = create_rnd_forrest(train_vec, train_set['Label'], 
                            random_state=42, max_leaf_nodes=100)
evaluate_algo(rndf, test_vec, test_set['Label'], train_vec, train_set['Label'])
#%% 
nb = create_nb(train_vec, train_set['Label'])
evaluate_algo(nb, test_vec, test_set['Label'], train_vec, train_set['Label'])



