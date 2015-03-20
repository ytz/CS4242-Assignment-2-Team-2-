from __future__ import division

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import csv

def main(output_as):
	train_file = "tweetsLIWC.csv"

	# Read training data
	train = pd.read_csv(train_file)
	test_file = "testTweetsPreprocessed.csv"
	test = pd.read_csv(test_file)
	test.fillna(0,inplace=True)
	train.fillna(0,inplace=True)
	

	#user_id = train["ID"]

	if output_as == 'age':
		#train_file = "tweetsLIWCage.csv"
		#train = pd.read_csv(train_file)
		#train.fillna(0,inplace=True)
		# Choose AGE 
		del train["ID"]
		del train["GENDER"]
		target = train["AGE"]
		del train["AGE"]

		del test["ID"]
		del test["GENDER"]
		test_target = test["AGE"]
		del test["AGE"]
	elif output_as == 'gender':
		#train_file = "tweetsLIWCgender.csv"
		#train = pd.read_csv(train_file)
		#train.fillna(0,inplace=True)
		# Choose GENDER
		del train["ID"]
		del train["AGE"]
		target = train["GENDER"]
		del train["GENDER"]

		del test["ID"]
		
		test_target = test["GENDER"]
		del test["GENDER"]
		del test["AGE"]
	

	features = train.as_matrix()
	test_feat = test.as_matrix()
	
	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	le = preprocessing.LabelEncoder()
	le.fit(test_target)
	test_target = le.transform(test_target)

	# Train Classifier
	print("Training the Classifier")

	"""
	*~* Pick your classifier here *~*
	"""
	#classifier = LinearSVC(class_weight='auto') # SVM
	#classifier = GaussianNB() # Naive Bayes
	#classifier = KNeighborsClassifier(n_neighbors=10) # KNN
	classifier = RandomForestClassifier()

	# Train Classifier
	classifier.fit(features, target)

	
	#predictions_train = classifier.predict(features)
	#accuracy = metrics.accuracy_score(target, predictions_train)
	#f1 = metrics.f1_score(target, predictions_train)

	predictions_train = classifier.predict(test_feat)
	accuracy = metrics.accuracy_score(test_target, predictions_train)
	f1 = metrics.f1_score(test_target, predictions_train)

	print output_as
	print "Accuracy: %f" % accuracy
	print "F1: %f" % f1

	predictions_train = le.inverse_transform(predictions_train)
	return predictions_train

	

predictions_age = main(output_as='age')
predictions_gender = main(output_as='gender')

header = ['ID','AGE','GENDER']
train = pd.read_csv("tweetsLIWCgender.csv")
user_id = train["ID"]
output = [user_id, predictions_age,predictions_gender]
output = zip(*output)
with open("tweets_output.csv", "wb") as f:
    writer = csv.writer(f)
    output.insert(0,header)
    writer.writerows(output)