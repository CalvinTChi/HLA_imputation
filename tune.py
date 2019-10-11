from keras.layers import Input, Dense, LSTM, Flatten, Dropout, Embedding, concatenate
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation, Input, Dropout, concatenate, TimeDistributed, Lambda
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from utils import *
from ConvNet import ConvNet
import keras.optimizers
import numpy as np
import pandas as pd
import pickle, os, sys, getopt, time

# hyperparameters to search from
embedding_dims = [4, 8, 16]
batch_sizes = [128, 256, 512]
n1_units = [64, 128, 256]
n2_units = [64, 128, 256]
n3_units = [32, 64, 128]
strides1 = [4, 8]
strides2 = [4, 8]
maxpools = [2, 3, 4]
dropouts = [0.5, 0.3, 0.1]

# other inputs
train_file = "../data/T1DGC_REF/train/T1DGC_REF_Train.txt"
result_file = "../results/T1DGC/ConvNet/tune.csv"
n_searches = 100
n_gram = 5
max_nb_words = 4**n_gram * 2

def main():
    train = pd.read_csv(train_file, delimiter=" ", header = 0)
    tokenizer = pickle.load(open("../models/T1DGC/ConvNet/tokenizer.p", "rb"))
    yEncoders = pickle.load(open("../models/T1DGC/ConvNet/yEncoders.p", "rb"))

    # generate n-grams
    max_seq_length = 0
    for i in range(8):
        train.iloc[:, i] = [generate_n_grams(list(s), n_gram) for s in train.iloc[:, i]]
        max_seq_length = max(max([len(s) for s in train.iloc[:, i]]), max_seq_length)

    # Map words and labels to numbers
    n_grams = []
    for i in range(8):
        n_grams += train.iloc[:, i].tolist()
    tokenizer = Tokenizer(num_words = max_nb_words)
    tokenizer.fit_on_texts(n_grams)

    trainY = train.iloc[:, 8:16]
    yEncoders = [LabelEncoder() for i in range(8)]
    for i in range(8):
        genename = trainY.columns[i]
        yEncoders[i].fit(trainY[genename])

    # Generate training, dev, and validation datasets
    train, validation = train_validation_split(train, p = 0.20)
    train, dev = train_validation_split(train, p = 0.15)
    trainX, trainY = generate_feature_label_pair(train, tokenizer, yEncoders, max_seq_length)
    devX, devY = generate_feature_label_pair(dev, tokenizer, yEncoders, max_seq_length)

    validationY = []
    validationX = np.zeros((validation.shape[0], 8, max_seq_length))
    for i in range(8):
        x = tokenizer.texts_to_sequences(validation.iloc[:, i])
        validationX[:, i, :] = pad_sequences(x, maxlen = max_seq_length, padding='post')
        genename = validation.columns[i + 8]
        validationY.append(validation[genename])

    # setup save file
    result_columns = ["embedding_dim", "batch_size", "n_1", "n_2", "n_3", "stride1", "stride2", 
        "maxpool", "dropout", "max_nb_words", "max_seq_length", "validation_accuracy"]
    save_df = pd.DataFrame(index = list(range(n_searches)), columns = result_columns)

    start_time = time.time()
    for i in range(n_searches):
        if i % 10 == 0:
            print("hyperparameter set {0}, {1} minutes".format(i, round((time.time() - start_time) / 60)))

        hyperparameters = {"embedding_dim": np.random.choice(embedding_dims, 1)[0],
                           "batch_size": np.random.choice(batch_sizes, 1)[0],
                           "n_1": np.random.choice(n1_units, 1)[0],
                           "n_2": np.random.choice(n2_units, 1)[0],
                           "n_3": np.random.choice(n3_units, 1)[0],
                           "stride1": np.random.choice(strides1, 1)[0],
                           "stride2": np.random.choice(strides2, 1)[0],
                           "maxpool": np.random.choice(maxpools, 1)[0],
                           "dropout": np.random.choice(dropouts, 1)[0],
                           "max_nb_words": max_nb_words,
                           "max_seq_length": max_seq_length,
                           "stride": 1}

        overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        model = ConvNet(yEncoders, hyperparameters)
        model.fit(trainX, trainY, epochs = 100, batch_size = hyperparameters["batch_size"], validation_data = (devX, devY), callbacks=[overfitCallback])
        predY = model.predict(validationX)

        classPred = []
        for j in range(len(validationY)):
            numPredY = np.argmax(predY[j], axis = 1)
            predYname = yEncoders[j].inverse_transform(numPredY)
            classPred.append(predYname)
        validation_accuracy = round(accuracy_score(validationY, classPred), 4)

        # record hyperparameters used
        save_df.loc[i, "embedding_dim"] = hyperparameters["embedding_dim"]
        save_df.loc[i, "batch_size"] = hyperparameters["batch_size"]
        save_df.loc[i, "n_1"] = hyperparameters["n_1"]
        save_df.loc[i, "n_2"] = hyperparameters["n_2"]
        save_df.loc[i, "n_3"] = hyperparameters["n_3"]
        save_df.loc[i, "stride1"] = hyperparameters["stride1"]
        save_df.loc[i, "stride2"] = hyperparameters["stride2"]
        save_df.loc[i, "maxpool"] = hyperparameters["maxpool"]
        save_df.loc[i, "dropout"] = hyperparameters["dropout"]
        save_df.loc[i, "max_nb_words"] = hyperparameters["max_nb_words"]
        save_df.loc[i, "max_seq_length"] = hyperparameters["max_seq_length"]
        save_df.loc[i, "validation_accuracy"] = validation_accuracy
    save_df.sort_values(by = ["validation_accuracy"], ascending = False, inplace = True)
    save_df.to_csv(result_file, index=False)

if __name__ == "__main__":
    main()


