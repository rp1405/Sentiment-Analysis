
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
import keras
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


ann=keras.models.load_model("SentimentPrediction.h5")
gensimModel=gensim.models.Word2Vec.load("GensimVocabModel.h5")


def sentenceToVector(sentence):
    result=[]
    for word in sentence:
        if word in gensimModel.wv:
            embedding = gensimModel.wv[word]
            result.append(embedding)
        else:
            result.append(np.random.uniform(-0.25, 0.25, gensimModel.vector_size))
    return result


def predictSentiment(sentence):
    sentence=gensim.utils.simple_preprocess(sentence)
    sentence=sentenceToVector(sentence)
    sentence=pad_sequences([sentence],maxlen=40,padding="post",dtype="float32")
    sentence=np.array(sentence,dtype="float32")
    sentence=sentence.flatten()
    sentence=np.array([sentence])
    result=ann.predict(sentence)[0][0]
    if result>.5:
        print("Positive: ",result)
    else:
        print("Negative: ",result)


statement=input("Enter Statement: ")
predictSentiment(statement)


