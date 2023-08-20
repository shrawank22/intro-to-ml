import numpy as np
from numpy import random as rand
from sklearn.linear_model import LogisticRegression
import pickle

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# PLEASE BE CAREFUL THAT ERROR CLASS NUMBERS START FROM 1 AND NOT 0. THUS, THE FIFTY ERROR CLASSES ARE
# NUMBERED AS 1 2 ... 50 AND NOT THE USUAL 0 1 ... 49. PLEASE ALSO NOTE THAT ERROR CLASSES 33, 36, 38
# NEVER APPEAR IN THE TRAINING SET NOR WILL THEY EVER APPEAR IN THE SECRET TEST SET (THEY ARE TOO RARE)

# Input Convention
# X: n x d matrix in csr_matrix format containing d-dim (sparse) bag-of-words features for n test data points
# k: the number of compiler error class guesses to be returned for each test data point in ranked order


# Output Convention
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of classes with the i-th row 
# containing k error classes which it thinks are most likely to be the correct error class for the i-th test point.
# Class numbers must be returned in ranked order i.e. the label yPred[i][0] must be the best guess for the error class
# for the i-th data point followed by yPred[i][1] and so on.

# CAUTION: Make sure that you return (yPred below) an n x k numpy nd-array and not a numpy/scipy/sparse matrix
# Thus, the returned matrix will always be a dense matrix. The evaluation code may misbehave and give unexpected
# results if an nd-array is not returned. Please be careful that classes are numbered from 1 to 50 and not 0 to 49.

def topFive(dic,k):
    res = []
    dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    c = k
    for i in dic:
        if c == 0:
            break
        res.append(i)
        c -= 1
    return np.array(res)

def findErrorClass( X, k ):
	pickled_model = pickle.load(open('model.pkl', 'rb'))
	y_pred_test = pickled_model.predict_proba(X)
	y_pred_fin = []
	for y in y_pred_test:
		j = 0
		y_pred_tmp = []
		for i in range(50):
			if i == 32 or i == 35 or i ==  37:
				y_pred_tmp.append(0)
			else:
				y_pred_tmp.append(y[j])
				j += 1
		y_pred_fin.append(y_pred_tmp)
	y_pred_fin = np.array(y_pred_fin)
	print(y_pred_fin.shape)
	res = []
	for y in y_pred_fin:
		y_dic = {}
		for i in range(50):
			y_dic[i+1] = y[i]
		r = topFive(y_dic,k)
		res.append(r)
	res = np.array(res)
	return res