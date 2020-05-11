'''
Points to note
1) Each firefly corresponds to a solution
2) The solution here is the best set of weights
ALGORITHM
steps:
	1.generate say n fireflies ( each is a random set of weights )
		xi=[w1,w2,w3.....wn] i=0 to total number of fireflies (n)
	2.Pass these weights to the nn and call mse ( objective fucntion or fitness function f(xi))
	3.Light intensity LIi at xi= 1/f(xi)
	4. Define light absorption coefficient 
	5. while (t <MaxGeneration)
	6. for i = 1 : n all n fireflies
	7. for j = 1 : i all n fireflies
	8. if (LIj >LIi), Move firefly i towards j in d-dimension
	9. end if
	10. Attractiveness varies with distance r via exp[−γr]
	11. Evaluate new solutions by varying the value of = [wi] by changing alpha with delta and update
	corresponding light intensity
	12. end for j
	13. end for i
	14. Rank the fireflies and find the current best (Li)
	15. end while
3) Algo returns the best set of weights to you
'''

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
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(filename = f'Results.log', filemode='w', format='%(asctime)s-%(message)s', level= logging.INFO )

def get_data():
	scaler = MinMaxScaler()
	data = pd.read_csv('haberman.csv')
	temp=data.values
	temp_scaled = scaler.fit_transform(temp)
	data = pd.DataFrame(temp_scaled)

	
	X=data.iloc[:,0:3]
	y=data[data.columns[-1]]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state =42)
	print("shape of input test-train split")
	print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))
	# y_train//=2
	# y_test//=2
	return X_train,X_test,y_train,y_test

def fitness_function(weights):
	weight_matrices=vector_to_weight(arch,weights)
	# weight_matrices=np.array(weight_matrices)
	original_matrix=model.get_weights()
	# print("Our weights...")
	# print(np.shape(weight_matrices))
	# print(weight_matrices)
	# print("keras weights...")
	# print(np.shape(original_matrix))
	# print(original_matrix)
	logging.info(".....................................................")
	model.set_weights(weight_matrices)
	logging.info(f"weights of firefly {weight_matrices}")
	# print("Keras weights have been changed....")
	# print(model.get_weights())
	X_train,X_test,y_train,y_test=get_data()
	opt= SGD(lr=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	train_loss,train_acc = model.evaluate(X_train,y_train)
	logging.info(f"Loss for this firefly {train_loss}")
	return train_loss

def sort_ff(x,li,n,dim):
	temp=np.zeros(dim)
	for i in range(0, (n- 1)):
			for j in range(0, (n-i-1)):
				if (li[j] < li[j+1]):
					z = li[j]  # exchange attractiveness
					li[j] = li[j+1]
					li[j+1] = z
					temp = x[j]  # exchange fitness
					x[j] = x[j+1]
					x[j+1] = temp

def ff(dimension,num_of_fireflies,epochs=10):
	#initialise parameters
	n=num_of_fireflies #Total Number of fireflies
	dim=dimension #Dimension of each firefly
	x=np.random.uniform(low=-0.5,high=0.5,size=(num_of_fireflies,dimension)) # Entire population of fireflies
	error=np.zeros(n)
	beta = 1 #always 1
	gamma = 1 #absorption coeff
	# alpha = random.uniform(0,1) #randomization parameter
	alpha = 0.1
	print("Initial Population\n",x)

	#finding the value objective function for each firefly
	for i in range(n):
		logging.info(f"----------firefly {i}---------")
		error[i]=fitness_function(x[i])
	print("Fitness Function value for each firefly is : \n",error)
	li=[0]*n #light intensity of each firefly li[i]=1/f(xi)

	for i in range(len(error)):
		li[i]=1/error[i]
	print("Light Intensity of population..\n",li)

	# Training
	t=0
	while(t < epochs):
		print(f"In epoch {t} of {epochs}")
		logging.info("#######################################################################################################")
		logging.info(f"In epoch {t} of {epochs}")
		for i in range(0,n):
			for j in range(0,i):
				if(li[j]>li[i]):
					rand=random.uniform(0,1)
					r=distance.euclidean(x[i],x[j])#cartesian distance
					x[i]=x[i]+beta*math.exp(-gamma * math.pow(r, 2.0))*(x[i]-x[j])+alpha*(rand-0.5) #updating x[i]'s

		#update li
		for i in range(n):
			logging.info(f"-------------------------------firefly {i}-----------------------------------------")
			error[i]=fitness_function(x[i])
		li=[0]*n #light intensity of each firefly li[i]=1/f(xi)

		for i in range(len(error)):
			li[i]=1/error[i]
		t=t+1

	#Sort fireflies based on li and pick best (highest li)
	print("Finished Training")
	sort_ff(x,li,n,dim)
	best = x[0]
	print("best weight vector...")
	print(best)
	print(type(best))
	return best

def weight_to_vector(arch):

	vector_dim=0
	for i in range(len(arch)-1):
		vector_dim+=arch[i]*arch[i+1]

	return vector_dim

def vector_to_weight(arch,weight_vector):

	num_of_matrices= len(arch)-1
	dim=[]
	sum_=0
	for i in range(len(arch)-1):
		sum_+=arch[i]*arch[i+1]
		dim.append(sum_)

	x=np.split(weight_vector,dim)
	#removing last empty list
	x=x[:-1]
	# print(x)
	# print(type(x),np.shape(x))
	weights=[]
	for i in range(num_of_matrices):
		weights.append(np.reshape(x[i],(arch[i],arch[i+1])))
		weights.append(np.zeros(arch[i+1],dtype=float))

		# print(x[i],np.shape(x[i]))

	# print("vector_to_weight",weights)
	# x has final weight matrices
	return weights






if __name__ == '__main__':
	X_train,X_test,y_train,y_test=get_data()
	model = Sequential()
	#input features = 3
	arch=[3,5,5,1]
	model.add(Dense(arch[1],input_dim=3, activation='relu'))
	model.add(Dense(arch[2], activation='relu'))
	model.add(Dense(arch[3], activation='sigmoid'))
	## ff algo
	num_of_fireflies= 10
	dimension=weight_to_vector(arch)
	model.summary()
	weight_vector=ff(dimension,num_of_fireflies,epochs=10)
	print("FINAL WEIGHT VECTOR...",weight_vector)
	final_weights=vector_to_weight(arch,weight_vector)

	model.set_weights(final_weights)
	print("Keras weights are set as")
	print(model.get_weights())
	print("--------------------FINAL TEST----------------------")
	opt = SGD(lr=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	test_loss,test_acc = model.evaluate(X_test,y_test)
	print("-----------------------------------------")
	print("test loss : ",test_loss)
	print("test accuracy : ",test_acc)
	print("-----------------------------------------")

	# model.fit(X, y, epochs=150, batch_size=10)
	# _, accuracy = model.evaluate(X, y)
	# print('Accuracy: %.2f' % (accuracy*100))

	# y_pred = model.predict(X_test)