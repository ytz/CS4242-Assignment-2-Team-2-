from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as dv
from sklearn_pandas import DataFrameMapper
import csv
from scipy import stats
import numpy as np

def main():
	dfTweets = pd.read_csv("tweets_output.csv")

	dfMedia = pd.read_csv("media_output.csv")
	dfCheck = pd.read_csv("checkin_output.csv")
	df = pd.merge(dfTweets, dfMedia, on='ID', how='outer')
	df.fillna('NA',inplace=True)
	df.columns = ['ID', 'ageT','genderT','ageM','genderM']
	
	df = pd.merge(df, dfCheck, on='ID', how='outer')
	df.fillna('NA',inplace=True)
	le = preprocessing.LabelEncoder()
	test = pd.read_csv("train.csv")
	df = pd.merge(df, test, on='ID', how='left')
	mapper = DataFrameMapper([ ('ageT',preprocessing.LabelBinarizer())])
	mapper.fit(df)
	mapper.transform(df)
	print df
	
	
	target = df["AGE"]
	del df["AGE"]
	del df["ID"]
	del df["GENDER"]

	


	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	dv.fit(df)
	features = dv.transform(df)
	
	# Train Classifier
	print("Training the Classifier")




	combinedClassifier = LogisticRegression()
	combinedClassifier.fit(features, target)
	predictions_train = combinedClassifier.predict(features)
	accuracy = metrics.accuracy_score(target, predictions_train)
	f1 = metrics.f1_score(target, predictions_train)

	print output_as
	print "Accuracy: %f" % accuracy
	print "F1: %f" % f1

	predictions_train = le.inverse_transform(predictions_train)
	return predictions_train





main()
