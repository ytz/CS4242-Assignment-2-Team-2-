import pandas as pd
import csv
from json import JSONDecoder
import json
import re

def main():
	n = 165508 # number of tweets
	data = []
	with open('tweets.json') as f:
		for line in f:
			line = line.decode('utf-8', 'ignore').encode('utf-8')
			
			data.append(json.loads(line))

	writer = csv.writer(open("tweets.csv", "wb+"))

# Write CSV Header, If you dont need that, remove this line
	#f.writerow(["pk", "model", "codename", "name", "content_type"])
	writer.writerow(["ID", "text"])
	for tweet in data:
		row = (
			tweet['userId'],
           	                    # tweet_id
            tweet['text'],            # tweet_time
            #tweet['creationDate'],   # tweet_author
                 # tweet_authod_id
            #tweet['hashtags'],                  # tweet_language
            #tweet['urls'],                   # tweet_geo
        )
		values = [(value.encode('utf8') if hasattr(value, 'encode') else value) for value in row]
		writer.writerow(values)

	# Merge with training data (ground truth)	
	df_train = pd.read_csv('train.csv')
	df_tweets = pd.read_csv('tweets.csv')	

	df_new = pd.merge(df_tweets, df_train, on='ID', how='left')
	df_new = df_new.fillna(0)

	df_new.to_csv('tweet_and_train.csv',index=False)



main()