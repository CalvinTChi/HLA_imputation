import numpy as np
import pandas as pd
import sys
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def train_validation_split(data, p = 0.15):
    data.sample(frac=1)
    n = data.shape[0]
    validationIdx = int(round(n * p))
    validation = data.iloc[0:validationIdx, :]
    train = data.iloc[validationIdx:n, :]
    return train, validation

def generate_feature_label_pair(data, tokenizer, yEncoders, max_seq_length):
    Y = []
    X = np.zeros((data.shape[0], 8, max_seq_length))
    for i in range(8):
        x = tokenizer.texts_to_sequences(data.iloc[:, i])
        X[:, i, :] = pad_sequences(x, maxlen = max_seq_length, padding='post')
        
        genename = data.columns[i + 8]
        y = yEncoders[i].transform(data[genename].tolist())
        y = to_categorical(y, num_classes = len(yEncoders[i].classes_))
        Y.append(y)
    return X, Y

def generate_n_grams(seq, n):
    grams = []
    gram_count = len(seq) // n
    for i in range(0, gram_count * n, n):
        grams.append("".join(seq[i:i + n]))
    return grams

def accuracy_score(classY, classPred):
    num_correct = 0
    for i in range(len(classY)):
        num_correct += np.sum(classY[i] == classPred[i])
    return float(num_correct) / (len(classY[0]) * 8)

# @param predY: list of predicted class as integers
# @param testY: list of true class as integers
# @yEncoder: sklearn.preprocessing.LabelEncoder is a mapping between class names and numbers
def calculate_recall(classY, classPred, yEncoders):
    classes = []
    for i in range(len(classY)):
        classes += list(yEncoders[i].classes_)
    recallDf = pd.DataFrame(0, index = classes,
        columns = ["number", "num_correct"])
    for i in range(len(classY)):
        for j in range(len(yEncoders[i].classes_)):
            allele = yEncoders[i].classes_[j]
            alleleIdx = np.where(classY[i] == allele)[0]
            recallDf.loc[allele, "number"] = len(alleleIdx)
            recallDf.loc[allele, "num_correct"] = len(np.intersect1d(alleleIdx, np.where(classPred[i] == allele)[0]))
    #recallDf["percent"] = recallDf["num_correct"] / recallDf["number"]
    recallDf = recallDf.sort_values(by = ["number"], ascending = False)
    return recallDf

    