from sklearn import svm
import argparse 
import json
from sklearn import preprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', help ='audio\'s vector for training')
parser.add_argument('--test_file', help = 'audio vectors for test')
parser.add_argument('--output_file',help = 'file to write results')
args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file

with open(train_file) as mon_fichier:
    data_train = json.load(mon_fichier)

with open(test_file) as mon_fichier:
    data_test = json.load(mon_fichier)

y_train = list(data_train.keys())
X_train = list(data_train.values())
print('shape:', len(X_train[1]))
y_test = list(data_test.keys())
X_test = list(data_test.values())

Y = []
for y in y_train:
	Y.append(y.split('/')[0])
Yt = []
for y in y_test:
	Yt.append(y.split('/')[0])

# le = preprocessing.LabelEncoder()
# Y_train = le.fit(Y)


clf = clf = svm.SVC()
clf.fit(X_train, Y)

y_pred = clf.predict(X_test)
print(y_pred)
print(Yt)





