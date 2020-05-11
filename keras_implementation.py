import math
import random
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from keras.models import Sequential
from keras.layers import Dense
import logging
from keras.optimizers import SGD

def get_data():
	data = pd.read_csv('haberman.csv')
	X=data.iloc[:,0:3]
	y=data[data.columns[-1]]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state =5)
	print("shape of input test-train split")
	print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))
	y_train//=2
	y_test//=2
	return X_train,X_test,y_train,y_test


if __name__ == '__main__':
	X_train,X_test,y_train,y_test=get_data()
	model = Sequential()
	#input features = 3
	arch=[3,2,2,1]
	model.add(Dense(arch[1],input_dim=3, activation='relu'))
	model.add(Dense(arch[2], activation='relu'))
	model.add(Dense(arch[3], activation='sigmoid'))
	## ff algo
	# ff_weights=[[[-0.42657104,  1.3147963 ],
	#    [-0.57474697,  2.1513243 ],
	#    [ 0.6965767 , -0.13519368]], 
	#    [0., 0.], 
	#    [[-1.7960453 ,  0.3873567 ],
	#    [-0.20379385, -0.73235995]], 
	#    [0., 0.], 
	#    [[0.1937068 ],
	#    [0.45154226]],
	#    [0.]]
	ff_weights=[[[0.2,0.2],
	   [0.2,  0.2 ],
	   [ 0.2 , 0.2]], 
	   [0., 0.], 
	   [[0.2 , 0.2 ],
	   [0.2, 0.2]], 
	   [0., 0.], 
	   [[0.2 ],
	   [0.2]],
	   [0.]]
	model.set_weights(ff_weights)
	print(model.get_weights())
	model.summary()
	opt = SGD(lr=0.002)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=3)
	loss, accuracy = model.evaluate(X_test, y_test)
	print('loss : ',loss)
	print('Accuracy: ',accuracy)