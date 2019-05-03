from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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

N_GRAM = 4
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

    model = load_model(model_directory + "/convnet0.h5")
    tokenizer = pickle.load(open(model_directory + "/tokenizer.p", "rb"))
    yEncoder = pickle.load(open(model_directory + "/yEncoder.p", "rb"))

    max_seq_length = int_shape(model.input)[1]

    test = pd.read_csv(test_file, delimiter=" ", header = 0)

    test['sequences'] = [list(s) for s in test['sequences']]
    test['sequences'] = [generate_n_grams(s, N_GRAM) for s in test['sequences']]

    testX, testY = generate_feature_label_pair(test, tokenizer, yEncoder, max_seq_length)

    recallDfs = []

    for i in range(B):
        print("bootstrap sample " + str(i + 1))
        sampleIdx = np.random.choice(testX.shape[0], replace=True, size = testX.shape[0])
    
        testXb = testX[sampleIdx, :]
        testYb = testY[sampleIdx, :]
        testYb = np.argmax(testYb, axis=1)
        
        predYb = model.predict_classes(testXb)
        
        recallDf = calculate_recall(predYb, testYb, yEncoder)
        recallDf.rename(columns = {'number':'number' + str(i + 1), 'num_correct': 'num_correct' + str(i + 1)}, 
                     inplace=True)
    
        recallDfs.append(recallDf)

    recallDf_all = pd.concat(recallDfs, axis = 1)
    
    # write result, along with performance on origin test dataset
    recallDf0 = pd.read_csv(result_directory + "/recall_by_allele0.csv", index_col = 0)
    recallDf_all = pd.merge(recallDf0, recallDf_all, left_index = True, right_index = True)
    recallDf_all.to_csv(result_directory + "/recall_by_alleleB.csv", index = True)

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: bootstrap.py -t <test_file> -m <model_directory> -r <result_directory>")
        sys.exit(2)
    main(sys.argv[1:])

