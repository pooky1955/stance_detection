import os
import pickle

def load_batch(filepaths,folder_prefix=None):
  """loads multiple pickle files"""
  loaded = []
  for filepath in filepaths:
    complete_filepath = f'{folder_prefix}/{filepath}.pickle'
    if folder_prefix == None:
      complete_filepath = f'{filepath}.pickle'
    with open(complete_filepath,"rb") as f:
      loaded_obj = pickle.load(f)
      loaded.append(loaded_obj)
  print("Loaded batches")
  return loaded

def dump_batch(filepaths,objects,folder_prefix=None):
  """Dumps multiple objects into its pickle file"""
  assert len(filepaths) == len(objects), "Lengths are not equal"
  if not os.path.exists(folder_prefix) and folder_prefix != None:
    os.mkdir(folder_prefix)
    print(f"Created directory : {folder_prefix}")

  for filepath, obj in zip(filepaths,objects):
    complete_filepath = f'{folder_prefix}/{filepath}.pickle'
    if folder_prefix == None:
      complete_filepath = f'{filepath}.pickle'

    with open(complete_filepath,"wb") as f:
      pickle.dump(obj,f)
  print("Dumped batches")