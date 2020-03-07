from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.backend import int_shape
import warnings
import keras.optimizers
from utils import *
import numpy as np
import pandas as pd
import pickle, os, sys, getopt
warnings.filterwarnings("ignore")

N_GRAM = 5
B = 1000

def main(argv):
    test_file = ""
    model_directory = ""
    result_directory = ""
    try:
        # i.e. "i:" means that argument for i is required
        opts, args = getopt.getopt(argv, "t:m:r:")
    except getopt.GetoptError:
        print("Usage: bootstrap.py -t <test_file> -m <model_directory> -r <result_directory>")
        # Exit status of 2 means command line synax error
        sys.exit(2)
    # opt = input identifier, arg = actual input

    for opt, arg in opts:
        if opt == "-t":
            test_file = arg
        elif opt == "-m":
            model_directory = arg
        elif opt == "-r":
            result_directory = arg

    model = load_model("../" + model_directory + "/convnet_tune.h5")
    tokenizer = pickle.load(open("../" + model_directory + "/tokenizer_tune.p", "rb"))
    yEncoders = pickle.load(open("../" + model_directory + "/yEncoders_tune.p", "rb"))

    max_seq_length = int_shape(model.input)[2]

    test = pd.read_csv("../" + test_file, delimiter=" ", header = 0)
    # generate n-grams
    for i in range(8):
        test.iloc[:, i] = [generate_n_grams(list(s), N_GRAM) for s in test.iloc[:, i]]

    # manipulate test dataset to be in the correct format
    testY = []
    testX = np.zeros((test.shape[0], 8, max_seq_length))
    for i in range(8):
        x = tokenizer.texts_to_sequences(test.iloc[:, i])
        testX[:, i, :] = pad_sequences(x, maxlen = max_seq_length, padding='post')
        genename = test.columns[i + 8]
        testY.append(test[genename])
    
    recallDfs = []
    recallDfB = pd.read_csv("../" + result_directory + "/recall_by_allele_tuned_0.csv", index_col = 0)
    recallDfB.to_csv("../" + result_directory + "/recall_by_alleleB_tuned.csv", index = True)

    for i in range(B):
        print("bootstrap sample " + str(i + 1))
        sampleIdx = np.random.choice(testX.shape[0], replace=True, size = testX.shape[0])
    
        testXb = testX[sampleIdx, :, :]
        testYb = [label[sampleIdx] for label in testY]
        predYb = model.predict(testXb)

        classPred = []
        for j in range(len(testY)):
            numPredY = np.argmax(predYb[j], axis = 1)
            predYname = yEncoders[j].inverse_transform(numPredY)
            classPred.append(predYname)
        
        recallDf = calculate_recall(testYb, classPred)
        recallDf.rename(columns = {'number':'number' + str(i + 1), 'number_correct': 'number_correct' + str(i + 1)}, 
                     inplace=True)
    
        recallDfs.append(recallDf)

        if (i + 1) % 100 == 0:
            recallDf_all = pd.concat(recallDfs, axis = 1)
            recallDfB = pd.read_csv("../" + result_directory + "/recall_by_alleleB_tuned.csv", index_col = 0)
            recallDf_all = pd.merge(recallDfB, recallDf_all, left_index = True, right_index = True)
            recallDf_all.to_csv("../" + result_directory + "/recall_by_alleleB_tuned.csv", index = True)
            recallDfs = []

    #recallDf_all = pd.concat(recallDfs, axis = 1)
    
    # write result, along with performance on origin test dataset
    #recallDf0 = pd.read_csv(result_directory + "/recall_by_allele0.csv", index_col = 0)
    #recallDf_all = pd.merge(recallDf0, recallDf_all, left_index = True, right_index = True)
    #recallDf_all.to_csv(result_directory + "/recall_by_alleleB.csv", index = True)

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: bootstrap.py -t <test_file> -m <model_directory> -r <result_directory>")
        sys.exit(2)
    main(sys.argv[1:])

