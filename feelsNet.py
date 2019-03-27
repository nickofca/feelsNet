#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing and embedding settings inspired by the Spacey net @ https://github.com/explosion/spacy/blob/master/examples/deep_learning_keras.py
Training requires the installation of this specific spacy model $ python -m spacy download en_vectors_web_lg
@author: nickofca
"""
import pandas as pd
import sklearn.metrics
import keras
import numpy as np
import spacy
import pickle
import sys

#Display full matrices from bash
np.set_printoptions(threshold=sys.maxsize)

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

#Tuning params
sent_len = 60
lr = 0.001
width = 128
epochs = 10
callbacks = [keras.callbacks.EarlyStopping(patience=2)]
threshold = .34 #.34 is max @0.568
predict_ = True
model_dir = "models/model_0.538.h5"
dynamic_threshold = False

class feelsNet(object):
    def __init__(self,model_dir="models/model_0.538.h5"):
        self.model = keras.models.load_model(model_dir)
        
    def train(self,train_data,dev_data):
        #Divide data
        train_text = train_data["Tweet"]
        train_labels = train_data[emotions].values
        dev_text = dev_data["Tweet"]
        dev_labels = dev_data[emotions].values
        
        #Spacy pipe initialization
        #"Hit 'em with the pipe!"
        nlp = spacy.load("en_vectors_web_lg")
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        embeddings = nlp.vocab.vectors.data
        train_docs, train_labels, train_x = self.postprocess(train_text,nlp,train_labels)
        dev_docs, dev_labels, dev_x = self.postprocess(dev_text,nlp,dev_labels)
        
        #Create keras model
        inputs = keras.Input((sent_len,))
        x = keras.layers.Embedding(embeddings.shape[0],embeddings.shape[1],
                                   input_length = sent_len, trainable = False,
                                   weights = [embeddings], mask_zero = True)(inputs)
        x = keras.layers.TimeDistributed(keras.layers.Dense(width))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(width))(x)
        x = keras.layers.Dense(11, activation = "sigmoid")(x)
        model = keras.Model(inputs = inputs, outputs = x)
        model.compile(keras.optimizers.Adam(lr = lr),loss = "mse")
        
        #Train model
        model.fit(train_x, train_labels, validation_data = (dev_x, dev_labels),
                  epochs = epochs, callbacks = callbacks)
        
        model.save("models/model.h5")
        self.model = model
        
        #For lists as input instead of csv
    def predictList(self,textList:list):
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        docs, x = self.postprocess(textList, nlp)
        pred_prob = self.model.predict(x)
        pred_int = np.where(pred_prob>threshold,1,0)
        output = pd.DataFrame(columns = ["Tweet"]+emotions)
        for i,tweet in enumerate(textList):
            output.loc[i*2] = [tweet]+pred_prob[i].tolist()
            output.loc[i*2+1] = [tweet]+pred_int[i].tolist()
        return output
    
    def predict(self,predict_data: list):
        predict_text = predict_data["Tweet"]
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        docs, x = self.postprocess(predict_text, nlp)
        return self.model.predict(x)
    
    def postprocess(self,text, nlp, labels=None):
        docs = list(nlp.pipe(text))
        x = self.get_features(docs, sent_len)
        if labels is None:
            return docs, x
        else:
            return docs, labels, x
        
    #Borrowed from Spacy tutorial
    def get_features(self,docs, max_length):
        docs = list(docs)
        Xs = np.zeros((len(docs), max_length), dtype='int32')
        for i, doc in enumerate(docs):
            j = 0
            for token in doc:
                vector_id = token.vocab.vectors.find(key=token.orth)
                if vector_id >= 0:
                    Xs[i, j] = vector_id
                else:
                    Xs[i, j] = 0
                j += 1
                if j >= max_length:
                    break
        return Xs
    
def quickTrain():
    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv("data/2018-E-c-En-train.txt", **read_csv_kwargs)
    pred_data = pd.read_csv("data/2018-E-c-En-dev.txt", **read_csv_kwargs)
    predictions_prob = pred_data.copy()
    predictions = pred_data.copy()
    
    # makes predictions on the dev set
    if predict_ == False:
        model1 = feelsNet().train(train_data, pred_data)
    else:
        model1 = feelsNet(model_dir)
    predictions_prob[emotions] = model1.predict(pred_data)

    # saves predictions and prints out multi-label accuracy
    if dynamic_threshold == True:
        with open("support/thresh_list.p","rb") as file:
            thresh_list = pickle.load(file)
        for i, emotion in enumerate(emotions):
            predictions[emotion] = predictions_prob[emotion].where(predictions_prob[emotion]>thresh_list[i],0)
            predictions[emotion] = predictions[emotion].where(predictions_prob[emotion]<=thresh_list[i],1)
    else:
            predictions[emotions] = predictions_prob[emotions].where(predictions_prob[emotions]>threshold,0)
            predictions[emotions] = predictions[emotions].where(predictions_prob[emotions]<=threshold,1).astype(int)
    predictions.to_csv("output/E-C_en_pred.txt", sep="\t", index=False)
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        pred_data[emotions], predictions[emotions])))
   
#grid search for optimal probability thresholds
def findThresh(predictions,groundTruth):
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
    scoreList = []
    thresholdList = []
    for emotion in emotions:
        best = [0,0]
        for i in range(2000):
            threshold = np.random.uniform()
            predictions[emotion] = predictions[emotion].where(predictions[emotion]>threshold,0)
            predictions[emotion] = predictions[emotion].where(predictions[emotion]<=threshold,1)
            score = sklearn.metrics.jaccard_similarity_score(groundTruth[emotion], predictions[emotion])
            if score > best[0]:
                best[0] = score
                best[1] = threshold
        scoreList.append(score)
        thresholdList.append(threshold)
    with open("support/thresh_list.p","wb") as file:
        pickle.dump(thresholdList,file)
    print(scoreList)

if __name__ == "__main__":
    #Call from command line with affect as argument and print out.
    if len(sys.argv) > 1:
        print(feelsNet().predictList([sys.argv[1]]))
    else:
        exampleOut = feelsNet().predictList(["How could company A do this to me?","Service was terrific for this place."])