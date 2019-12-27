# stance_detection
ML part for Expo-Science 2019-2020

Model trained on google colab using deep learning architecture.

Dataset from fnc-1 (Fake News Challenge 1 2017)

`fnc-1` : Initial folder of the competition fnc-1

`loader.py` : Transforms the initial csv files from fnc-1 into headlines, bodies, and one_hot_stances

`preprocess.py` : Preprocesses the dataset, cleans the data

`token_vectors.pickle` : Pickled dictionnary mapping most common words in the training dataset to google's Word2Vec 300 vectors

`train_deep` : Code used to train my deep learning model on Google Colab

