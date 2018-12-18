import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn import preprocessing


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


train_label=[
[0, 7, 0, 7, 0, 7, 0, 7],
[0, 3, 0, 7, 0, 7, 0, 7],
[4, 7, 0, 7, 0, 7, 0, 7],
[0, 2, 0, 7, 0, 7, 0, 7],
[3, 5, 0, 7, 0, 7, 0, 7],
[6, 7, 0, 7, 0, 7, 0, 7],
[0, 9, 0, 9, 0, 9, 0, 9],
[0, 4, 0, 9, 0, 9, 0, 9],
[5, 9, 0, 9, 0, 9, 0, 9],
[0, 3, 0, 9, 0, 9, 0, 9],
[4, 6, 0, 9, 0, 9, 0, 9],
[7, 9, 0, 9, 0, 9, 0, 9],
[0, 11, 0, 11, 0, 11, 0, 11],
[0, 5, 0, 11, 0, 11, 0, 11],
[6, 11, 0, 11, 0, 11, 0, 11],
[0, 3, 0, 11, 0, 11, 0, 11],
[4, 7, 0, 11, 0, 11, 0, 11],
[8, 11, 0, 11, 0, 11, 0, 11],
[0, 13, 0, 13, 0, 13, 0, 13],
[0, 6, 0, 13, 0, 13, 0, 13],
[7, 13, 0, 13, 0, 13, 0, 13],
[0, 4, 0, 13, 0, 13, 0, 13],
[5, 9, 0, 13, 0, 13, 0, 13],
[10, 13, 0, 13, 0, 13, 0, 13],
[0, 15, 0, 15, 0, 15, 0, 15],
[0, 7, 0, 15, 0, 15, 0, 15],
[8, 15, 0, 15, 0, 15, 0, 15],
[0, 5, 0, 15, 0, 15, 0, 15],
[6, 10, 0, 15, 0, 15, 0, 15],
[11, 15, 0, 15, 0, 15, 0, 15],
[0, 17, 0, 17, 0, 17, 0, 17],
[0, 8, 0, 17, 0, 17, 0, 17],
[9, 17, 0, 17, 0, 17, 0, 17],
[0, 5, 0, 17, 0, 17, 0, 17],
[6, 11, 0, 17, 0, 17, 0, 17],
[12, 17, 0, 17, 0, 17, 0, 17],
[0, 19, 0, 19, 0, 19, 0, 19],
[0, 9, 0, 19, 0, 19, 0, 19],
[10, 19, 0, 19, 0, 19, 0, 19],
[0, 6, 0, 19, 0, 19, 0, 19],
[7, 13, 0, 19, 0, 19, 0, 19],
[14, 19, 0, 19, 0, 19, 0, 19],
[0, 21, 0, 21, 0, 21, 0, 21],
[0, 10, 0, 21, 0, 21, 0, 21],
[11, 21, 0, 21, 0, 21, 0, 21],
[0, 7, 0, 21, 0, 21, 0, 21],
[8, 14, 0, 21, 0, 21, 0, 21],
[15, 21, 0, 21, 0, 21, 0, 21],
[0, 23, 0, 23, 0, 23, 0, 23],
[0, 11, 0, 23, 0, 23, 0, 23],
[12, 23, 0, 23, 0, 23, 0, 23],
[0, 7, 0, 23, 0, 23, 0, 23],
[8, 15, 0, 23, 0, 23, 0, 23],
[16, 23, 0, 23, 0, 23, 0, 23],
[0, 25, 0, 25, 0, 25, 0, 25],
[0, 12, 0, 25, 0, 25, 0, 25],
[13, 25, 0, 25, 0, 25, 0, 25],
[0, 8, 0, 25, 0, 25, 0, 25],
[9, 17, 0, 25, 0, 25, 0, 25],
[18, 25, 0, 25, 0, 25, 0, 25],
[0, 27, 0, 27, 0, 27, 0, 27],
[0, 13, 0, 27, 0, 27, 0, 27],
[14, 27, 0, 27, 0, 27, 0, 27],
[0, 9, 0, 27, 0, 27, 0, 27],
[10, 18, 0, 27, 0, 27, 0, 27],
[19, 27, 0, 27, 0, 27, 0, 27],
[0, 29, 0, 29, 0, 29, 0, 29],
[0, 14, 0, 29, 0, 29, 0, 29],
[15, 29, 0, 29, 0, 29, 0, 29],
[0, 9, 0, 29, 0, 29, 0, 29],
[10, 19, 0, 29, 0, 29, 0, 29],
[20, 29, 0, 29, 0, 29, 0, 29]
]
train_set=[
[0, 7, 0, 7],
[0, 4, 0, 7],
[3, 7, 0, 7],
[0, 3, 0, 7],
[2, 6, 0, 7],
[5, 7, 0, 7],
[0, 9, 0, 9],
[0, 5, 0, 9],
[4, 9, 0, 9],
[0, 4, 0, 9],
[3, 7, 0, 9],
[6, 9, 0, 9],
[0, 11, 0, 11],
[0, 6, 0, 11],
[5, 11, 0, 11],
[0, 4, 0, 11],
[3, 8, 0, 11],
[7, 11, 0, 11],
[0, 13, 0, 13],
[0, 7, 0, 13],
[6, 13, 0, 13],
[0, 5, 0, 13],
[4, 10, 0, 13],
[9, 13, 0, 13],
[0, 15, 0, 15],
[0, 8, 0, 15],
[7, 15, 0, 15],
[0, 6, 0, 15],
[5, 11, 0, 15],
[10, 15, 0, 15],
[0, 17, 0, 17],
[0, 9, 0, 17],
[8, 17, 0, 17],
[0, 6, 0, 17],
[5, 12, 0, 17],
[11, 17, 0, 17],
[0, 19, 0, 19],
[0, 10, 0, 19],
[9, 19, 0, 19],
[0, 7, 0, 19],
[6, 14, 0, 19],
[13, 19, 0, 19],
[0, 21, 0, 21],
[0, 11, 0, 21],
[10, 21, 0, 21],
[0, 8, 0, 21],
[7, 15, 0, 21],
[14, 21, 0, 21],
[0, 23, 0, 23],
[0, 12, 0, 23],
[11, 23, 0, 23],
[0, 8, 0, 23],
[7, 16, 0, 23],
[15, 23, 0, 23],
[0, 25, 0, 25],
[0, 13, 0, 25],
[12, 25, 0, 25],
[0, 9, 0, 25],
[8, 18, 0, 25],
[17, 25, 0, 25],
[0, 27, 0, 27],
[0, 14, 0, 27],
[13, 27, 0, 27],
[0, 10, 0, 27],
[9, 19, 0, 27],
[18, 27, 0, 27],
[0, 29, 0, 29],
[0, 15, 0, 29],
[14, 29, 0, 29],
[0, 10, 0, 29],
[9, 20, 0, 29],
[19, 29, 0, 29]
]

#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

#ax_arr = (ax1, ax2)
print("Size of train label " + str(len(train_label)))
print("Size of train set " + str(len(train_set)))


X_train = np.array(train_label)
Y_train = np.array(train_set)

#plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 1], s=50, cmap='autumn');

#print(Y_train[:, 0])


#in comments here are the values I have tested
n_estimators =500 #decrease the value to get worse results ? 500 is giving good result already
max_depth=4
min_samples_split=2
learning_rate=0.01
loss='ls' #least square loss



clf_0 = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,min_samples_split=min_samples_split,max_depth=max_depth,n_estimators=n_estimators)
clf_0.fit(X_train, Y_train[:, 0])

clf_1 = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,min_samples_split=min_samples_split,max_depth=max_depth,n_estimators=n_estimators)
clf_1.fit(X_train, Y_train[:, 1])

clf_2 = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,min_samples_split=min_samples_split,max_depth=max_depth,n_estimators=n_estimators)
clf_2.fit(X_train, Y_train[:, 2])

clf_3 = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,min_samples_split=min_samples_split,max_depth=max_depth,n_estimators=n_estimators)
clf_3.fit(X_train, Y_train[:, 3])


print("Expected :")
print(train_set[-1])

print("Results")
predict_0 = clf_0.predict([[20, 29, 0, 29, 0, 29, 0, 29]])
print(predict_0 )

predict_1 = clf_1.predict([[20, 29, 0, 29, 0, 29, 0, 29]])
print(predict_1 )

predict_2 = clf_2.predict([[20, 29, 0, 29, 0, 29, 0, 29]])
print(predict_2 )

predict_3 = clf_3.predict([[20, 29, 0, 29, 0, 29, 0, 29]])
print(predict_3 )


fig,ax = plt.subplots()
currAxis= plt.gca()

currAxis.add_patch( plt.Rectangle( (19, 29), 1, -1, alpha=1, facecolor='coral') )
plt.text(19, 29,'Expected')

currAxis.add_patch( plt.Rectangle( (predict_0, predict_1), (predict_2-predict_0), -(predict_1-predict_3), alpha=1, facecolor='lightblue'))
plt.text(predict_0, predict_1,'Obtained')

plt.ylim(-5, 50)
plt.xlim(-5, 55)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Obtained VS Expected')
plt.rc('grid', linestyle="-.", color='lightgrey')
plt.grid(True)
plt.show()




#saving classifier to disk with scikit learn :
#from joblib import dump, load
#dump(clf, 'filename.joblib')
#clf = load('filename.joblib')


