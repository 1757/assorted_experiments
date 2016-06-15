from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv as csv
import numpy as np
import re
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords



#model.save_word2vec_format('GG.txt', binary=False)
vectorizer = CountVectorizer(min_df=1)

#Reading
csv_file_object = csv.reader(open('Sheettfidf.csv', 'rb'))
csv_file_object.next()

#Data Cleaning
data = []
regex = re.compile('[^a-zA-Z0-9\s"\n"]')
regex_num_space = re.compile('[0-9]+[abcdefghijklmnopqruvwxyz]')
regex_multiple = re.compile('[\s][\s]+')

for row in csv_file_object:
	v = re.sub(regex, ' ', row[0])
	v = re.sub(regex_multiple, ' ', v)
	data.append(v)

data = np.array(data)



cachedStopWords = stopwords.words("english")
wordlists = []



for sentences in data:
	text = [word for word in sentences.split() if word not in cachedStopWords]
	wordlists.append(text)

#-------------MLP
import sknn
import csv
import numpy as np
import re

from sknn.mlp import Classifier, Layer


def count_correct_one(predict, real, index):
	count = 0
	count_one = 0
	print predict.shape
	print real.shape
	for i in range(len(predict[index])):
		if real[index][i]==1:
			count_one +=1
			if str(predict[index][i]) == str(real[index][i]):
				count +=1
	try:
		toreturn = float(count)/ float(count_one)
	except:
		toreturn = None
	return toreturn

def count_correct_zero(predict, real, index):
	count = 0
	count_one = 0
	for i in range(len(predict[index])):
		if real[index][i]==0:
			count_one +=1
			if str(predict[index][i]) == str(real[index][i]):
				count +=1
	try:
		toreturn = float(count)/ float(count_one)
	except:
		toreturn = None
	return toreturn


def count_correct(predict, real, index):
	count = 0
	print predict.shape
	print real.shape
	for i in range(len(predict[index])):
		if str(predict[index][i]) == str(real[index][i]):
			count +=1
	return float(count)/ float(len(predict[index]))




y = [[] for i in range(5)]

with open('Combined.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	y = [[] for i in range(5)]
	for row in reader:
		#print row
		for j in range(len(row)):
			if row[j] == '1':
				y[j].append(1)
			else:
				y[j].append(0)








y = np.asarray(y)
#print y
y = y.T



randList = np.random.permutation(4087)

#############-------------TFIDF---------
vectorizer = TfidfVectorizer(min_df = 1)
x_data  = vectorizer.fit_transform(data).toarray()

xtrain = []
ytrain = []
xtest  = []
ytest  = []

ytrain1 = []
ytest1 = []	



for i in range(0, 3000):
	xtrain.append(x_data[randList[i]])
	ytrain.append(y[randList[i]])
	ytrain1.append([y[randList[i]][1]])

for i in range(3000, len(y)):
	xtest.append(x_data[randList[i]])
	ytest.append(y[randList[i]])
	ytest1.append([y[randList[i]][1]])



xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)

xtest = np.asarray(xtest)
ytest = np.asarray(ytest)

ytrain1 = np.asarray(ytrain1)
ytest1 = np.asarray(ytest1)
print ytrain[1].shape

#print xtest[1]

##------EXPERIMENTS:

# print "Experiment 1 starts"


# unitList = [100, 200, 300, 500, 700]

# for unitNumber in range(len(unitList)):
# 	print "result for %s nodes" %unitList[unitNumber]

# 	nn = Classifier(
# 	    layers=[
# 	        Layer("Rectifier", units = unitList[unitNumber]),
# 	        Layer("Softmax")],
# 	    dropout_rate = 0.0,
# 	    learning_rate=0.001,
# 	    loss_type = 'mcc',
# 	    n_iter=25)

# 	nn.fit(xtrain, ytrain)

# 	ypredict = nn.predict(xtest) #) for i in range(len(xtest))]



# 	mm = [count_correct(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]
# 	mm1 = [count_correct_one(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]
# 	mm2 = [count_correct_zero(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]

# 	print mm
# 	print mm1
# 	print mm2

# 	count = 0
# 	for i in range(len(ytest)):
# 		if ytest[i].tolist() == ypredict[i].tolist():
# 			print ytest[i].tolist()
# 			print ypredict[i].tolist()
# 			count +=1
# 	print float(count)/len(ytest)

print "Experiment 1 starts"


unitList = [0.0, 0.1, 0.25, 0.4, 0.6]

for unitNumber in range(len(unitList)):
	print "result for %s dropout" %unitList[unitNumber]

	nn = Classifier(
	    layers=[
	        Layer("Rectifier", units = 300),
	        Layer("Softmax")],
	    dropout_rate = unitList[unitNumber],
	    learning_rate=0.001,
	    loss_type = 'mcc',
	    n_iter=25)

	nn.fit(xtrain, ytrain)

	ypredict = nn.predict(xtest) #) for i in range(len(xtest))]



	mm = [count_correct(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]
	mm1 = [count_correct_one(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]
	mm2 = [count_correct_zero(ypredict.T, ytest.T, i) for i in range(len(ypredict.T))]

	print mm
	print mm1
	print mm2

	count = 0
	for i in range(len(ytest)):
		if ytest[i].tolist() == ypredict[i].tolist():
			#print ytest[i].tolist()
			#print ypredict[i].tolist()
			count +=1
	print float(count)/len(ytest)


