import pandas as pd
import csv
from json import JSONDecoder
import json
import re
import datetime
from datetime import timedelta
from sklearn import feature_extraction


def test():
	checkin = json.load(open('json\checkins.json'), cls=ConcatJSONDecoder, encoding="ISO-8859-1")
	date_time = datetime.datetime.utcfromtimestamp(checkin[0]['createdAt'])
	local_date_time = date_time + timedelta(minutes=checkin[0]['timeZoneOffset'])
	print local_date_time
	print local_date_time.hour

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
	header_list = ['ID', 'hereNow', 'checkinsCount', 'userCount', 'tipCount', 'hour', 'month']

	for idx, myCheckin in enumerate(checkin):
		userid_data = checkin[idx]['userId']
		row = []
		try:
			row.append(myCheckin['venue']['hereNow']['count'])
			row.append(myCheckin['venue']['stats']['checkinsCount'])
			row.append(myCheckin['venue']['stats']['usersCount'])
			row.append(myCheckin['venue']['stats']['tipCount'])

			date_time = datetime.datetime.utcfromtimestamp(myCheckin['createdAt'])
			local_date_time = date_time + timedelta(minutes=myCheckin['timeZoneOffset'])

			if (local_date_time.hour < 4):
				hour = '0-3'
			elif (local_date_time.hour < 8):
				hour = '4-7'
			elif (local_date_time.hour < 12):
				hour = '8-11'
			elif (local_date_time.hour < 16):
				hour = '12-15'
			elif (local_date_time.hour < 20):
				hour = '16-19'
			elif (local_date_time.hour < 24):
				hour = '20-23'

			row.append(hour)
			row.append('month_'+str(local_date_time.month))

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

	# Temporarily remove string col for normalisation
	user_id = df_checkin['ID']
	hour_col = df_checkin['hour']
	month_col = df_checkin['month']
	del df_checkin['ID']
	del df_checkin['hour']
	del df_checkin['month']

	# normalise
	print df_checkin.dtypes
	df_checkin = df_checkin.fillna(0)
	df_checkin = (df_checkin - df_checkin.mean()) / (df_checkin.max() - df_checkin.min())
	
	# Add back user_id
	df_checkin.insert(0,'ID',user_id)
	df_checkin.insert(1,'hour',hour_col)
	df_checkin.insert(2,'month',month_col)

	# One-hot encoding 'hour' and 'month'
	df_checkin, _= one_hot_dataframe(df_checkin, ['hour'], replace=True)
	df_checkin, _= one_hot_dataframe(df_checkin, ['month'], replace=True)
	del df_checkin['hour']
	del df_checkin['month']

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


def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
        Modified from https://gist.github.com/kljensen/5452382
    """
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

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
#test()