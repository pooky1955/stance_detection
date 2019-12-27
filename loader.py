import pandas as pd
import argparse
import os
import pickle
from helpers import dump_batch
from tqdm import tqdm
def load_csvs(filepaths):
  """loads csv files as pandas Dataframe"""
  loaded_csvs = []
  for filepath in filepaths:
    loaded_csv = pd.read_csv(filepath)
    loaded_csvs.append(loaded_csv)
  return loaded_csvs

  

all_stances = ["agree","disagree","discuss","unrelated"]

def one_hot(label,all_labels):
  """One hot encodes a label given all labels"""
  one_hot_arr = [0 for _ in range(len(all_labels))]
  for i,label_i in enumerate(all_labels):
    if label == label_i:
      one_hot_arr[i] = 1
  return one_hot_arr

def preprocess(df_stances,df_bodies):
  """Processed both dfs and returns lists of bodies, headlines, and one-hot encoded labels"""
  bodies = []
  headlines = []
  stances = []
  one_hot_stances = []
  df_body_indexed = df_bodies.set_index("Body ID")
  for headline,bodyId,stance in tqdm(df_stances.values):
    headlines.append(headline)
    body = df_body_indexed.loc[bodyId].articleBody
    bodies.append(body)
    stances.append(stance)
    one_hot_stance = one_hot(stance,all_stances)
    one_hot_stances.append(one_hot_stance)
  return headlines,bodies,one_hot_stances,stances

if __name__ == "__main__":
  bodies_path = input("Enter path to bodies df")
  stances_path = input("Enter path to stances df")

  df_bodies, df_stances = load_csvs([bodies_path,stances_path])

  print("Stances distribution :")

  print(df_stances.groupby("Stance").size())
  headlines,bodies,one_hot_stances,stances = preprocess(df_stances,df_bodies)

  output_folder = input("Enter folder name for output directory")
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    print(f"Creating folder {output_folder}")
    
  dump_batch(["headlines","bodies","one_hot_stances","stances"],objects=[headlines,bodies,one_hot_stances,stances],folder_prefix=output_folder)