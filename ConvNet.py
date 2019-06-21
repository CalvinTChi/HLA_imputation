from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation, Input, Dropout, concatenate, TimeDistributed, Lambda
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from utils import *
import keras.optimizers
import numpy as np
import pandas as pd
import pickle, os, sys, getopt
#warnings.filterwarnings("ignore")

BATCH_SIZE = 256
EMBEDDING_DIM = 10
N_GRAM = 5
MAX_NB_WORDS = 4**N_GRAM * 2

def ConvNet(yEncoders, max_nb_words, embedding_dim = 10, max_seq_length = 500):
    main_input = Input(shape = (8, max_seq_length), name = "main_input")
    x = TimeDistributed(Embedding(input_dim = max_nb_words + 1,
                                  output_dim = embedding_dim,
                                  input_length = max_seq_length,
                                  trainable = True))(main_input)
    
    # convolution 1st layer
    x = TimeDistributed(Conv1D(128, 10, activation = 'relu', input_shape = (embedding_dim, 1)))(x)
    x = TimeDistributed(MaxPooling1D(3))(x)
    x = TimeDistributed(BatchNormalization())(x)
    
    # convolution 2nd layer
    x = TimeDistributed(Conv1D(64, 5, activation = 'relu'))(x)
    x = TimeDistributed(MaxPooling1D(3))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Flatten())(x)
    
    dense1 = [Lambda(lambda x, ind: x[:, ind, :], arguments={'ind': i})(x) for i in range(8)]
    
    dense2 = []
    for i in range(8):
        if i == 0:
            dense2.append(concatenate([dense1[0], dense1[1]]))
        elif i == 7:
            dense2.append(concatenate([dense1[6], dense1[7]]))
        else:
            dense2.append(concatenate([dense1[i - 1], dense1[i], dense1[i + 1]]))
    
    dense3 = []
    for i in range(8):
        dense3.append(Dense(64, activation = 'relu')(dense2[i]))
    
    outputs = []
    for i in range(8):
        outputs.append(Dense(len(yEncoders[i].classes_), activation = 'softmax')(dense3[i]))
    
    model = Model(inputs = main_input, outputs = outputs)
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics=['accuracy'])
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

    train = pd.read_csv(train_file, delimiter=" ", header = 0)
    test = pd.read_csv(test_file, delimiter=" ", header = 0)

    # generate n-grams
    MAX_SEQ_LENGTH = 0
    for i in range(8):
        train.iloc[:, i] = [generate_n_grams(list(s), N_GRAM) for s in train.iloc[:, i]]
        test.iloc[:, i] = [generate_n_grams(list(s), N_GRAM) for s in test.iloc[:, i]]
        MAX_SEQ_LENGTH = max(max([len(s) for s in train.iloc[:, i]]), MAX_SEQ_LENGTH)

    # Map words and labels to numbers
    n_grams = []
    for i in range(8):
        n_grams += train.iloc[:, i].tolist()
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(n_grams)
    pickle.dump(tokenizer, open(model_output + "/tokenizer.p", "wb"), protocol = 2)

    trainY = train.iloc[:, 8:16]
    yEncoders = [LabelEncoder() for i in range(8)]
    for i in range(8):
        genename = trainY.columns[i]
        yEncoders[i].fit(trainY[genename])
    pickle.dump(yEncoders, open(model_output + "/yEncoders.p", "wb"), protocol = 2)
    
    # Generate training and test datasets
    train, validation = train_validation_split(train, p = 0.10)
    trainX, trainY = generate_feature_label_pair(train, tokenizer, yEncoders, MAX_SEQ_LENGTH)
    validationX, validationY = generate_feature_label_pair(validation, tokenizer, yEncoders, MAX_SEQ_LENGTH)
    
    testY = []
    testX = np.zeros((test.shape[0], 8, MAX_SEQ_LENGTH))
    for i in range(8):
        x = tokenizer.texts_to_sequences(test.iloc[:, i])
        testX[:, i, :] = pad_sequences(x, maxlen = MAX_SEQ_LENGTH, padding='post')
        genename = test.columns[i + 8]
        testY.append(test[genename])

    overfitCallback = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=1,
                                verbose=0, mode='auto')
    model = ConvNet(yEncoders, max_nb_words = MAX_NB_WORDS, embedding_dim = 10, max_seq_length = MAX_SEQ_LENGTH)
    model.fit(trainX, trainY, epochs = 100, batch_size = BATCH_SIZE, 
          validation_data = (validationX, validationY), callbacks=[overfitCallback])

    predY = model.predict(testX)

    # evaluate on test set
    classPred = []
    for i in range(len(testY)):
        numPredY = np.argmax(predY[i], axis = 1)
        predYname = yEncoders[i].inverse_transform(numPredY)
        classPred.append(predYname)
    
    print("Test accuracy: ", round(accuracy_score(classY, classPred), 4))

    recallDf = calculate_recall(predY, testY, yEncoders)
    recallDf.to_csv(result_output + "/recall_by_allele0.csv", index = True)

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: ConvNet.py -t <train_file> -v <test_file> -m <model directory> -r <result directory>")
        sys.exit(2)
    main(sys.argv[1:])

