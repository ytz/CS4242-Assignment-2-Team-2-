import pandas as pd
import csv
from json import JSONDecoder
import json
import re

"""
Output media_and_train.csv which contains 
image features and ground truth
"""
def main():
	n = 4113 # number of images
	features_list = []
	header_list = ['ID']
	# Read files
	for x in range(1, n+1):
		filename = '.\\im_feature\\' + str(x) + '.txt'
		textfile = open(filename,'r')
		long_string = textfile.read()
		features = long_string.split()	# delimit by space
		features_list.append(features)

		header_list.append('f'+str(x))	# append feature no.

	# Get userID from media.json
	media = json.load(open('..\json\media.json'), cls=ConcatJSONDecoder, encoding="ISO-8859-1")

	for idx, myMedia in enumerate(media):
		number = idx + 1
		userid_data = media[idx]['userId']
		features_list[idx].insert(0,userid_data)

	features_list.insert(0,header_list)

	# Output csv with image feature list + userID
	with open("media.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(features_list)

	df_train = pd.read_csv('train.csv')
	df_media = pd.read_csv('media.csv')

	# Merge with training data (ground truth)
	df_new = pd.merge(df_media, df_train, on='ID', how='left')
	df_new = df_new.fillna(0)

	df_new.to_csv('media_and_train.csv',index=False)

"""
http://stackoverflow.com/questions/8730119/retrieving-json-objects-from-a-text-file-using-python
"""
#shameless copy paste from json/decoder.py
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)

class ConcatJSONDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)

        objs = []
        end = 0
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs

main()