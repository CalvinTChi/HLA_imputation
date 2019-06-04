import numpy as np
import pandas as pd
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

# @param predY: list of predicted class as integers
# @param testY: list of true class as integers
# @yEncoder: sklearn.preprocessing.LabelEncoder is a mapping between class names and numbers
def calculate_recall(predY, testY, yEncoder):
    recallDf = pd.DataFrame(0, index = yEncoder.classes_,
        columns = ["number", "num_correct"])
    testYname = yEncoder.inverse_transform(testY)
    predYname = yEncoder.inverse_transform(predY)
    for i in range(recallDf.shape[0]):
        allele = yEncoder.classes_[i]
        alleleIdx = np.where(testYname == allele)[0]
        recallDf.loc[allele, "number"] = len(alleleIdx)
        recallDf.loc[allele, "num_correct"] = len(np.intersect1d(alleleIdx, np.where(predYname == allele)[0]))
    #recallDf["percent"] = recallDf["num_correct"] / recallDf["number"]
    recallDf = recallDf.sort_values(by = ["number"], ascending = False)
    return recallDf