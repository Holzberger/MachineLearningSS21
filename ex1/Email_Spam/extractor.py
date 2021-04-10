import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

import pickle
import os.path
from os import path

import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def pickle_data(data_item=[], link_folder="./", name="data", operation="wb"):
    with open(link_folder+name, operation) as fp:
        if operation =="wb":
            pickle.dump(data_item, fp)
        if operation == "rb":
            return pickle.load(fp)
         
        
def import_dataset(link_folder="./", name_dataset="dataset.csv"):
    return pd.read_csv(link_folder+name_dataset)

def import_email_data(link_folder="./"):
    enorm_spam_data = import_dataset(link_folder, "enronSpamSubset.csv")
    ling_spam_data  = import_dataset(link_folder, "lingSpam.csv")
    
    # drop some meaningless cols
    enorm_spam_data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    ling_spam_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # drop last rwo since it summarizes all rows again
    enorm_spam_data.drop(enorm_spam_data.tail(1).index, inplace=True) 
    
    return pd.concat([enorm_spam_data, ling_spam_data], ignore_index=True)


def remove_duplicates(dataset, remove_dups=True, print_dups=False):
    if print_dups:
        print("There are {} duplicates.".format(np.sum(dataset.duplicated())))
    if remove_dups:
        return dataset.drop_duplicates() 
        

def remove_missing_vals(dataset, remove_missing=True, print_missing=False):
    if print_missing:
        print("There are:\n {} \n missing values.".format(dataset.isna().sum()))
    if remove_missing:
        return dataset.dropna()


def prep_mails(text_col, features=[1,1,1,1,1]):
    
    text_col = [text[7:] for text in text_col] # do not include "Subject:" prefix
    
    if features[0]:
        # remove links since they end up in tokens with no meaning
        text_col = [re.sub(r'http\S+', '', text) for text in text_col]
    
    if features[1]:
        # remove everything exept alphabetical characters and numbers
        pattern = ["[^a-zA-Z0-9]", "[^a-zA-Z]"]
        text_col = [re.sub(pattern[1]," ",text) for text in text_col]
    
    if features[2]:
        # convert uppercase chars in lowercase chars
        text_col = [text.lower() for text in text_col]
    
    # turn sentences into seperate worlds
    data_tokenized = [nltk.word_tokenize(text) for text in text_col]
    
    if features[3]:
        # lemmatize all world, that is convert them into most basic form
        lemma = WordNetLemmatizer()
        data_tokenized = [[lemma.lemmatize(word) for word in text] for text in data_tokenized]
    
    if features[4]:
        stopwords = nltk.corpus.stopwords.words("english")
        data_tokenized = [[word for word in text if word not in stopwords] for text in data_tokenized]
    
    return [" ".join(text) for text in data_tokenized]