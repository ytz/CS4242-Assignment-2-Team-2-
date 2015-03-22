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
import os

def preprocess(df, copy=False):
    df.fillna("",inplace=True)
    #porter_stemmer = PorterStemmer()
    #api = twitter_data.getAPI()
    print len(df.index)
    # Iterate tweets
    x = 1
    #y = 1
    users = []
    userIndex ={}
    for index,row in df.iterrows():
            # Retrieve user for a particular row
            user = row["ID"]
            if(user not in users):
                users.append(user)
                fileName = 'C:\Users\Dell user\Desktop\School\Year 4 Sem 2\CS4242\LIWCtweets/testUsers\LIWCtext'+str(x)+'.txt'
                f = open(fileName, 'w')
                userIndex[user] = x
                x= x+1
            else: 
                y = userIndex.get(user)
                fileName = 'C:\Users\Dell user\Desktop\School\Year 4 Sem 2\CS4242\LIWCtweets/testUsers\LIWCtext'+str(y)+'.txt'
                f = open(fileName, 'a')
            #Retrieve tweet for a particular row

            tweet = row['text']
            data ="\r\n%s" % tweet
            f.write(data)
            f.close()
    fileName = 'C:\Users\Dell user\Desktop\School\Year 4 Sem 2\CS4242\LIWCtweets/testusers.txt' 
    f = open(fileName, 'w')       
    for u in users:          
        data = u +"\n"
        f.write(data)
    f.close()
    return df

def main():
    df_train = pd.read_csv('tweet_and_trainZeros.csv')
    #df_clean = df_train.drop_duplicates(cols=['text'])
   
    #print len(df_train.index)
    df_train = preprocess(df_train)
    #print len(df_train.columns)
   
    #df_train.to_csv("preprocess_tweet.csv", na_rep="0",index=False,encoding='utf-8')
    
    #misc 
    #df_train = pd.read_csv('CS4242-Assignment 2 test (structure).csv')
    #df_tweets = pd.read_csv('tweet_and_trainZeros.csv') 

    #df_new = pd.merge(df_tweets, df_train, on='ID', how='left')
    #df_new = df_new.fillna(0)

    #df_new.to_csv('tweet_and_trainZeros.csv',index=False)

main()