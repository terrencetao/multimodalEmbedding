from sklearn import svm
import argparse 
import json
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

le = preprocessing.LabelEncoder()
le.fit(Yt)
y_true = le.transform(Yt)


clf = clf = svm.SVC()
clf.fit(X_train, Y)

y_pred = clf.predict(X_test)
y_view = le.transform(y_pred)
# print(y_view)
# print(Yt)
print(classification_report(y_true, y_view, target_names=le.classes_))
print(accuracy_score(y_true, y_pred))






