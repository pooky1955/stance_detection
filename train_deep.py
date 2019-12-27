from train_w2vec import encode_docs
import numpy as np
from helpers import load_batch, dump_batch
from constants import  HEADLINES_MAX_LEN, BODIES_MAX_LEN, NUM_DIMENSIONS_WORD2VEC
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Input, Flatten, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils import plot_model
import pickle
from tqdm import tqdm

def load_token_vectors(filepath):
  with open(filepath,"rb") as f:
    return pickle.load(f)

def encode_docs(documents,token_vectors,num_dimensions_word2vec,max_len):
  """Encodes a text document so that the model can use it as input"""
  vectors_sequences = np.zeros(shape=(len(documents),max_len,num_dimensions_word2vec),dtype="float16")
  for document_ind,document in tqdm(enumerate(documents)):
    tokens = document.split()
    for i in range(min(max_len,len(tokens))):
      token = tokens[i]
      if token in token_vectors:
        vectors_sequences[document_ind,i] = token_vectors[token]
  
  return vectors_sequences

def build_model(headlines_max_len,bodies_max_len,num_dimensions_word2vec):
  num_features = 100
  kernel_size_1 = 8
  kernel_size_2 = 6

  stride_size_1 = 1
  stride_size_2 = 1

  input_headlines = Input(shape=(headlines_max_len,num_dimensions_word2vec))
  conv_headlines_1_1 = Conv1D(2*2*num_features,kernel_size_1,kernel_regularizer=l2(l=0.01),strides=stride_size_1,activation="relu")(input_headlines)
  conv_headlines_1_2 = Conv1D(2*2*num_features,kernel_size_2,strides=stride_size_2,activation="relu")(conv_headlines_1_1)
  max_pool_headlines_1_3 = MaxPool1D()(conv_headlines_1_2)


  headlines_flatten = Flatten()(max_pool_headlines_1_3)


  input_bodies = Input(shape=(bodies_max_len,num_dimensions_word2vec))

  conv_bodies_1 = Conv1D(2*2*num_features,kernel_size_1,activation="relu",strides=stride_size_1)(input_bodies)
  conv_bodies_2 = Conv1D(2*2*num_features,kernel_size_2,strides=stride_size_2,activation="relu")(conv_bodies_1)
  max_pool_bodies_3 = MaxPool1D()(conv_bodies_2)

  conv_bodies_2_1 = Conv1D(2*num_features,8,activation="relu")(max_pool_bodies_3)
  conv_bodies_2_2 = Conv1D(2*num_features,6,activation="relu")(conv_bodies_2_1)
  max_pool_bodies_2_3 = MaxPool1D()(conv_bodies_2_2)


  conv_bodies_3_1 = Conv1D(num_features,8,activation="relu")(max_pool_bodies_2_3)
  conv_bodies_3_2 = Conv1D(num_features,6,activation="relu")(conv_bodies_3_1)
  max_pool_bodies_3_3 = MaxPool1D()(conv_bodies_3_2)

  bodies_flatten = Flatten()(max_pool_bodies_3_3)

  concat = concatenate([headlines_flatten,bodies_flatten])

  output = Dense(4,activation="softmax")(concat) 

  model = Model(inputs=[input_headlines,input_bodies],outputs=output)
  model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["categorical_accuracy"])
  model.summary()
  plot_model(model,to_file="model.png",show_shapes=True)

  return model

def main():
  filepath_to_token_vectors = "token_vectors.pickle"
  token_vectors = load_token_vectors(filepath_to_token_vectors)

  data_folder_path = "training_data"
  headlines, bodies , one_hot_stances = load_batch(["headlines","bodies","one_hot_stances"],folder_prefix=data_folder_path)

  vec_headlines = encode_docs(headlines,token_vectors,NUM_DIMENSIONS_WORD2VEC,HEADLINES_MAX_LEN)

  vec_bodies = encode_docs(bodies,token_vectors,NUM_DIMENSIONS_WORD2VEC,BODIES_MAX_LEN)  

  model = build_model(HEADLINES_MAX_LEN,BODIES_MAX_LEN,NUM_DIMENSIONS_WORD2VEC)
  print("Headlines shape")
  print(vec_headlines.shape)
  print("Bodies shape")
  print(vec_bodies.shape)

  # dump_batch(["vec_headlines","vec_bodies","one_hot_stances"],[vec_headlines,vec_bodies,one_hot_stances],folder_prefix="google_colab")


  print("Finished")
if __name__ == "__main__":
  main()