from __future__ import division
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tag import pos_tag 
import pandas as pd
import re
from nltk.util import ngrams
import itertools

def preprocess(df, copy=False):
    df.fillna("",inplace=True)
    #porter_stemmer = PorterStemmer()
    #api = twitter_data.getAPI()
    print len(df.index)
    # Iterate tweets
    x = 1
    y = 1
    for index,row in df.iterrows():
            # Retrieve tweet for a particular row
            tweet = row['text']
            fileName = 'C:\Users\Dell user\Desktop\School\Year 4 Sem 2\CS4242\Assignment2\LIWCtweets/tweet'+str(x)+'\LIWCtext' + str(y)+".txt"
            f = open(fileName, 'w')
            f.write(tweet)
            f.close()
            y=y+1
            if(y==451):
                x=x+1
                y=1

    return df

def main():
    df_train = pd.read_csv('tweets_age.csv')
    #df_clean = df_train.drop_duplicates(cols=['text'])
   
    #print len(df_train.index)
    df_train = preprocess(df_train)
    #print len(df_train.columns)
   
    #df_train.to_csv("preprocess_tweet.csv", na_rep="0",index=False,encoding='utf-8')

main()