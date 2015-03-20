import pandas as pd
import csv
from json import JSONDecoder
import json
import re


def preprocess():
	# Get info from checkin.json
	checkin = json.load(open('json\checkins.json'), cls=ConcatJSONDecoder, encoding="ISO-8859-1")
	"""
	print checkin[0]['userId']
	print checkin[0]['venue']['categories'][0]['name']
	print len(checkin[0]['venue']['categories'])
	print checkin[0]['venue']['hereNow']['count']
	print checkin[0]['venue']['stats']['checkinsCount']
	print checkin[0]['venue']['stats']['usersCount']
	print checkin[0]['venue']['stats']['tipCount']
	"""

	features_list = []
	header_list = ['ID', 'hereNow', 'checkinsCount', 'userCount', 'tipCount']

	for idx, myCheckin in enumerate(checkin):
		userid_data = checkin[idx]['userId']
		row = []
		try:
			row.append(myCheckin['venue']['hereNow']['count'])
			row.append(myCheckin['venue']['stats']['checkinsCount'])
			row.append(myCheckin['venue']['stats']['usersCount'])
			row.append(myCheckin['venue']['stats']['tipCount'])
		except KeyError:
			row = [0,0,0,0]

		row.insert(0,userid_data)

		features_list.append(row)

	features_list.insert(0, header_list)

	train_file = 'misc/train.csv'
	df_train = pd.read_csv(train_file)
	df_checkin = pd.DataFrame(features_list)
	# 1st row as column header
	df_checkin.columns = df_checkin.iloc[0]
	df_checkin = df_checkin.reindex(df_checkin.index.drop(0))

	# Temporarily remove user_id for normalisation
	user_id = df_checkin['ID']
	del df_checkin['ID']

	# normalise
	print df_checkin.dtypes
	df_checkin = df_checkin.fillna(0)
	df_checkin = (df_checkin - df_checkin.mean()) / (df_checkin.max() - df_checkin.min())
	
	# Add back user_id
	df_checkin.insert(0,'ID',user_id)

	# Add categorical information (1-hot encoding)
	for idx, myCheckin in enumerate(checkin):
		try:
			for x in range(len(myCheckin['venue']['categories'])):
				cat = myCheckin['venue']['categories'][x]['name']
				try:
					if (isinstance( df_checkin.loc[idx, cat], int)):
						df_checkin.loc[idx, cat] = 1
					else:
						df_checkin.loc[idx, cat] = 0
				except KeyError:
					df_checkin[cat] = 0
					df_checkin.loc[idx, cat] = 1
		except:
			continue

	# Merge with training data (ground truth)
	df_new = pd.merge(df_checkin, df_train, on='ID', how='left')
	df_new = df_new.fillna(0)
	df_new = df_new[ (df_new.AGE != 0) & (df_new.GENDER != 0)]

	df_new.to_csv('checkin_and_train.csv',index=False,encoding='utf-8')



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

preprocess()