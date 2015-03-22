from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as dv
from sklearn_pandas import DataFrameMapper
import csv
from scipy import stats
import numpy as np

def main(output_as):
	dfTweets = pd.read_csv("tweets_output.csv")

	dfMedia = pd.read_csv("media_output.csv")
	dfCheck = pd.read_csv("checkin_output.csv")
	df = pd.merge(dfTweets, dfMedia, on='ID', how='outer')
	df.fillna('NA',inplace=True)
	df.columns = ['ID', 'ageT','genderT','ageM','genderM']
	
	df = pd.merge(df, dfCheck, on='ID', how='outer')
	df.columns = ['ID', 'ageT','genderT','ageM','genderM','ageC','genderC']
	df.fillna('NA',inplace=True)
	le = preprocessing.LabelEncoder()
	lb = preprocessing.LabelBinarizer()
	
	test = pd.read_csv("train.csv")
	df = pd.merge(df, test, on='ID', how='left')

	if output_as == 'gender':
		target = df["GENDER"]
		del df["AGE"]
		del df["ID"]
		del df["GENDER"]
		#del df["ageT"]
		#del df["ageM"]
	
		mapper = DataFrameMapper([ ('ageT',lb),('genderT',le),('ageM',lb),('genderM',le),('ageC',lb), ('genderC',le)]) #didn't include ageC because of encoding issues
		mapper.fit(df)
		df = mapper.transform(df)
		print df.shape
		le.fit(target)
		target = le.transform(target)
	elif output_as == 'age':
		target = df["AGE"]
		del df["AGE"]
		del df["ID"]
		del df["GENDER"]
		#del df["genderT"]
		#del df["genderM"]
		#del df["genderC"]
		mapper = DataFrameMapper([('ageT',lb),('genderT',le),('ageM',lb),('genderM',le), ('ageC',lb),('genderC',le) ]) #didn't include ageC because of encoding issues
		mapper.fit(df)
		df = mapper.transform(df)
		
		lb.fit(target)
		target = lb.transform(target)


	#le = preprocessing.LabelEncoder()
	

	features = df
	
	# Train Classifier
	print("Training the Classifier")




	combinedClassifier = RandomForestClassifier()
	combinedClassifier.fit(features, target)
	predictions_train = combinedClassifier.predict(features)
	

	precision = metrics.precision_score(target, predictions_train, average='macro')
	recall = metrics.recall_score(target, predictions_train)
	accuracy = metrics.accuracy_score(target, predictions_train)
	f1 = metrics.f1_score(target, predictions_train)


	print output_as
	print "Precision: %f" % precision
	print "recall: %f" % recall
	print "Accuracy: %f" % accuracy
	print "F1: %f" % f1

	if output_as == 'gender':
		predictions_train = le.inverse_transform(predictions_train)
	elif output_as == 'age':	
		predictions_train = lb.inverse_transform(predictions_train)
	return predictions_train


predictions_age = main(output_as='age')
predictions_gender = main(output_as='gender')

header = ['ID','AGE','GENDER']
train = pd.read_csv("testTweetsEveryonePreprocessed.csv")
user_id = train["ID"]
output = [user_id, predictions_age,predictions_gender]
output = zip(*output)
with open("combined_output.csv", "wb") as f:
    writer = csv.writer(f)
    output.insert(0,header)
    writer.writerows(output)

# Get rid of duplicates
df = pd.read_csv("combined_output.csv")

for index, row in df.iterrows():
	row_id = row['ID']
	df_sub = df[ df.ID == row_id ]
	if df_sub.shape[0] != 1:
		new_age = df_sub['AGE'].value_counts().idxmax()
		if index == 0:
			print new_age
		new_gender = df_sub['GENDER'].value_counts().idxmax()\

		df = df[ df.ID != row_id ]
		df.loc[index] = [row_id, new_age, new_gender]
	else:
		continue

df.to_csv("combined_output.csv",index=False)
