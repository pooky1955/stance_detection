import pickle
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import string
import re
from nltk.corpus import stopwords
from helpers import load_batch, dump_batch


def join_tokens(tokens):
  """Joins tokens together as a string"""
  return ' '.join(tokens)

def tokenize(document,lower=False):
  """Cleans a document and returns its tokenized form"""
  words = [word for word in document.split()]
  if lower:
    words = [lower(word) for word in words]
  re_punc = re.compile(f'[{string.punctuation}]')
  words = [re_punc.sub("",word) for word in words]
  stopwords_set = set(stopwords.words("english"))
  words = [word for word in words if not word in stopwords_set]
  words = [word for word in words if len(word) > 1]
  return words


def add_vocab(tokens,vocab_counter):
  """adds document vocabulary to global vocabulary"""
  vocab_counter.update(tokens)


def save_list(list_to_save,filepath):
  """saves list of tokens"""
  text = '\n'.join(list(list_to_save))
  with open(filepath,"w",encoding="utf-8") as f:
    f.write(text)


def filter_tokens(document,tokens):
  """filters document only if in tokens"""
  filtered = [word for word in document if word in tokens]
  return filtered

def process_related(headlines,bodies,one_hot_stances):
  clean_related_bodies, clean_related_headlines, clean_related_one_hot_stances = [],[],[]
  
  unrelated_one_hot_stances = np.array(one_hot_stances)[:,3]
  for i , unrelated_one_hot_stance in enumerate(unrelated_one_hot_stances):
    if unrelated_one_hot_stance == 0:
      clean_related_bodies.append(bodies[i])  
      clean_related_headlines.append(headlines[i])
        
      clean_related_one_hot_stances.append(one_hot_stances[i][:3])  

  return clean_related_headlines, clean_related_bodies, clean_related_one_hot_stances

def main():
  """Preprocessing texts"""
  print("Preprocessing...")
  #splitting train test split
  headlines, bodies, one_hot_stances,stances = load_batch(["headlines","bodies","one_hot_stances","stances"],folder_prefix="loaded_data")
  headlines_train, headlines_test, bodies_train, bodies_test, one_hot_stances_train,one_hot_stances_test, stances_train, stances_test = train_test_split(headlines,bodies,one_hot_stances,stances,random_state=42,test_size=0.2) 
  print("Finished train-test split")

  #tokenizing dataset
  tokens_bodies_train = [tokenize(document) for document in bodies_train]
  tokens_headlines_train = [tokenize(document) for document in headlines_train]
  print("Tokenized training dataset")

  #adding to vocabulary counter
  vocab_counter = Counter()
  [add_vocab(tokens,vocab_counter) for tokens in tokens_bodies_train]
  [add_vocab(tokens,vocab_counter) for tokens in tokens_headlines_train]
  print("Vocabulary done")

  #filtering tokens and saving
  min_occurence = 3
  tokens = [token for token,count in dict(vocab_counter).items() if count > min_occurence]
  save_list(tokens,"tokens.txt")
  print("Tokens saved")

  #using these tokens to filter again the dataset
  tokens = set(tokens)
  clean_tokens_bodies_train = [filter_tokens(document,tokens) for document in tokens_bodies_train]
  clean_tokens_headlines_train = [filter_tokens(document,tokens) for document in tokens_headlines_train]
  print("Cleaned training dataset")

  #now joining tokens together
  clean_bodies_train = [join_tokens(tokens) for tokens in clean_tokens_bodies_train]
  clean_headlines_train = [join_tokens(tokens) for tokens in clean_tokens_headlines_train]

  chunk_size = 1000
  print("Finished creating training dataset")

                
  dump_batch(["bodies","headlines","stances","one_hot_stances"],[clean_bodies_train,clean_headlines_train,stances_train,one_hot_stances_train],folder_prefix="training_data")

  """Make the test folder"""
  tokens_bodies_test = [tokenize(document) for document in bodies_test]
  tokens_headlines_test = [tokenize(document) for document in headlines_test]
  print("Tokenized test dataset")

  clean_tokens_bodies_test = [filter_tokens(document,tokens) for document in tokens_bodies_test]
  clean_tokens_headlines_test = [filter_tokens(document,tokens) for document in tokens_headlines_test]
  print("Cleaned test dataset")

  clean_bodies_test = [join_tokens(tokens) for tokens in clean_tokens_bodies_test]
  clean_headlines_test = [join_tokens(tokens) for tokens in clean_tokens_headlines_test]
  print("Finished creating testing dataset")

  dump_batch(["bodies","headlines","one_hot_stances","stances"],[clean_bodies_test,clean_headlines_test,one_hot_stances_test,stances_test],folder_prefix="testing_data")

  clean_related_headlines_train, clean_related_bodies_train, clean_related_one_hot_stances_train = process_related(clean_headlines_train,clean_bodies_train,one_hot_stances_train)
  clean_related_headlines_test, clean_related_bodies_test, clean_related_one_hot_stances_test = process_related(clean_headlines_test,clean_bodies_test,one_hot_stances_test)
  dump_batch(["bodies","headlines","one_hot_stances"],[clean_related_bodies_train,clean_related_headlines_train,clean_related_one_hot_stances_train],folder_prefix="training_data/related")
  dump_batch(["bodies","headlines","one_hot_stances"],[clean_related_bodies_test,clean_related_headlines_test,clean_related_one_hot_stances_test],folder_prefix="testing_data/related")
  print("Finished creating unrelated dataset")
if __name__ == "__main__":
  main()