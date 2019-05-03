from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
import keras.optimizers
import warnings
from utils import *
import numpy as np
import pandas as pd
import pickle, os, sys, getopt
warnings.filterwarnings("ignore")

N_GRAM = 4
EMBEDDING_DIM = 25
BATCH_SIZE = 1024
epochs = 3

def create_model(n_classes, embedding_dim = 25, n_gram = 4, max_seq_length = 800):
    max_nb_words = 4**n_gram
    embedding_layer = Embedding(max_nb_words + 1,
                                embedding_dim,
                                input_length = max_seq_length,
                                trainable = True)
    model = Sequential()
    model.add(embedding_layer)
    # convolution 1st layer
    model.add(Conv1D(126, 10, activation = 'relu', input_shape = (embedding_dim, 1)))
    model.add(MaxPooling1D(3))
    model.add(BatchNormalization())
    
    # convolution 2nd layer
    model.add(Conv1D(126, 5, activation = 'relu'))
    model.add(MaxPooling1D(3))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(126, activation = 'relu'))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])
    return model

def main(argv):
    train_file = ""
    test_file = ""
    model_output = ""
    result_output = ""
    try:
        # i.e. "i:" means that argument for i is required
        opts, args = getopt.getopt(argv, "t:v:m:r:")
    except getopt.GetoptError:
        print("Usage: ConvNet.py -t <train_file> -v <test_file> -m <model directory> -r <result directory>")
        # Exit status of 2 means command line synax error
        sys.exit(2)
    # opt = input identifier, arg = actual input

    for opt, arg in opts:
        if opt == "-t":
            train_file = arg
        elif opt == "-v":
            test_file = arg
        elif opt == "-m":
            model_output = arg
        elif opt == "-r":
            result_output = arg

    # create output directories if do not exist
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    if not os.path.exists(result_output):
        os.makedirs(result_output)

    train = pd.read_table(train_file, delimiter=" ", header = 0)
    test = pd.read_table(test_file, delimiter=" ", header = 0)

    # preprocess the data
    data = pd.concat([train, test])

    train['sequences'] = [list(s) for s in train['sequences']]
    train['sequences'] = [generate_n_grams(s, N_GRAM) for s in train['sequences']]

    test['sequences'] = [list(s) for s in test['sequences']]
    test['sequences'] = [generate_n_grams(s, N_GRAM) for s in test['sequences']]

    MAX_SEQ_LENGTH = max([len(s) for s in train['sequences']])
    
    yEncoder = LabelEncoder()
    yEncoder = yEncoder.fit(list(data['hla']))
    pickle.dump(yEncoder, open(model_output + "/yEncoder.p", "wb"), protocol = 2)

    tokenizer = Tokenizer(num_words = 4**N_GRAM)
    tokenizer.fit_on_texts(train.iloc[:, 0])
    pickle.dump(tokenizer, open(model_output + "/tokenizer.p", "wb"), protocol = 2)

    trainX, trainY = generate_feature_label_pair(train, tokenizer, yEncoder, MAX_SEQ_LENGTH)
    testX, testY = generate_feature_label_pair(test, tokenizer, yEncoder, MAX_SEQ_LENGTH)

    # train model
    model = create_model(n_classes = len(yEncoder.classes_), embedding_dim = EMBEDDING_DIM, n_gram = N_GRAM, max_seq_length = MAX_SEQ_LENGTH)
    model.fit(trainX, trainY, epochs = epochs, batch_size = BATCH_SIZE)
    model.save(model_output + "/convnet0.h5")

    # evaluate on test set
    testY = np.argmax(testY, axis = 1)
    predY = model.predict_classes(testX)
    print("Test accuracy: ", round(accuracy_score(testY, predY), 4))
    recallDf = calculate_recall(predY, testY, yEncoder)
    recallDf.to_csv(result_output + "/recall_by_allele0.csv", index = True)

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: ConvNet.py -t <train_file> -v <test_file> -m <model directory> -r <result directory>")
        sys.exit(2)
    main(sys.argv[1:])


