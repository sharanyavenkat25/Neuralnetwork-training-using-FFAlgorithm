import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import random
from datetime import datetime
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import logging

#seed value to generate the same random matrix at all times
np.random.seed(26)
logging.basicConfig(filename = f'Results.log', filemode='w', format='%(process)d-%(levelname)s-%(asctime)s-%(message)s', level= logging.INFO )

class NN:
	"""
	The neural network class has instances for initialising weight parameters,
	Activation fucntions and their respective derivatives(defined sigmoid and ReLU),
	Train function with feed forward and back propogation using Stichastic Gradient 
	Descent Algorithm and a predict function which predicts a test set based on the trained 
	parameters

	"""
	def __init__(self, Ip, Hidden, Op):
		self.Ip = Ip
		self.Hidden= Hidden
		self.Op = Op

		self.lr = 0.1
		
		self.weightsIH = np.random.rand(self.Ip,self.Hidden) * 2 - 1
		self.weightsHO = np.random.rand(self.Hidden,self.Op) * 2 - 1
		
		# self.biasH = np.random.rand(self.Hidden) * 2 - 1
		# self.biasO = np.random.rand(self.Op) * 2 - 1
	
	def sigmoid(self, x, w):
		z = np.dot(x, w)
		return 1/(1 + np.exp(-z))

	def sigmoid_derivative(self, x, w):
		return self.sigmoid(x, w) * (1 - self.sigmoid(x, w))

	def relu(self, x, w):
		z=np.dot(x,w)
		for i in range(0, len(z)):
			for k in range(0, len(z[i])):
				if z[i][k] >=0:
					pass  
				else:
					z[i][k] = 0
		return z

	def relu_derivative(self, x, w):
		z=np.dot(x,w)
		for i in range(0, len(z)):
			for k in range(len(z[i])):
				if z[i][k] >=0:
					z[i][k] = 1
				else:
					z[i][k] = 0
		return z

	def feed_forward(self,X):

		self.z=self.relu(X,self.weightsIH)
		y=self.sigmoid(self.z,self.weightsHO)
		return y

	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat i.e the observed values

		"""
		y=self.feed_forward(X)
		return y
		

	
	def train(self,x,y,epochs):

		"""
		The training happens with a simple feed forward of parameters followed by calculation
		of the error (yhat-y); yhat is the predicted value and y is the actual value
		the error is then backpropogated to calucate gradients for the weight matrix
		(g_wjk & g_wij are the gradients calculated for the weight matrices
		weightsHO and weightsIH respectively which stand for the weight matrices between
		Hidden and output layer & Input and hidden layer)

		Returns the yhat

		"""	
		self.l=[]
		self.epochs=[]
		self.error=[]
		for i in range(epochs):
			
			Xi = x
			Xj = self.relu(Xi,self.weightsIH)
			yhat = self.sigmoid(Xj, self.weightsHO)

			e=abs(np.mean(np.square(y-yhat)))
			
			# gradients for hidden to output weights
			g_wHO = np.dot(Xj.T, (y - yhat) * self.sigmoid_derivative(Xj, self.weightsHO))
			# gradients for input to hidden weights
			g_wIH = np.dot(Xi.T, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.weightsHO), self.weightsHO.T) * self.relu_derivative(Xi, self.weightsIH))
			# update weights
			self.weightsIH +=self.lr*g_wIH
			self.weightsHO +=self.lr*g_wHO
			"""
			lr decay or lr scheduler ( time based decay )
			lr=initial lr*(1/1+decay*number of iterations)

			"""
			self.lr=self.lr*(1/(1+0.001*epochs))
			
			self.l.append(self.lr)
			self.epochs.append(i)
			self.error.append(e)

		return yhat



def data_cleaning(data):
	"""
	The following function is used to clean the noisy data in the dataset
	It replaces missing values with median/mode accordingly
	"""
	median1 = data['age'].median()
	data['age'].fillna(median1, inplace=True)
	median2 = data['BP1'].median()
	data['BP1'].fillna(median2, inplace=True)
	mean1 = data['weight1'].mean()
	data['weight1'].fillna(mean1, inplace=True)
	mode1 = data['education'].mode()
	data['education'].fillna(mode1[0], inplace=True)
	mode2 = data['res'].mode()
	data['res'].fillna(mode2[0], inplace=True)
	mode3 = data['history'].mode()
	data['history'].fillna(mode3[0], inplace=True)
	median3 = data['HB'].median()
	data['HB'].fillna(median3, inplace=True)
	X_features=data[["community","age","weight1","history","HB","IFA",'BP1',"education","res"]]
	Y_labels=data[["reslt"]]

	return X_features,Y_labels

def CM(y_test,y_test_obs):
	for i in range(len(y_test_obs)):
		if(y_test_obs[i]>0.6):
			y_test_obs[i]=1
		else:
			y_test_obs[i]=0
	
	cm=[[0,0],[0,0]]
	fp=0
	fn=0
	tp=0
	tn=0
	
	for i in range(len(y_test)):
		if(y_test[i]==1 and y_test_obs[i]==1):
			tp=tp+1
		if(y_test[i]==0 and y_test_obs[i]==0):
			tn=tn+1
		if(y_test[i]==1 and y_test_obs[i]==0):
			fp=fp+1
		if(y_test[i]==0 and y_test_obs[i]==1):
			fn=fn+1
	cm[0][0]=tn
	cm[0][1]=fp
	cm[1][0]=fn
	cm[1][1]=tp

	p= tp/(tp+fp)
	r=tp/(tp+fn)
	f1=(2*p*r)/(p+r)

	return cm,p,r,f1



if __name__ == '__main__':
	#call an instance of the neural network class
	nn=NN(9,10,1)
	epochs=20
	#load data, clean noisy data, split as test and train
	data = pd.read_csv('Andhra_dataset2.csv')
	X,y=data_cleaning(data)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state =42)
	X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
	
	#Normalise data values
	sc = SC()
	X_train = sc.fit_transform(X_train)
	X_test = sc.fit_transform(X_test)
	print(f"\n##########	Neural Network to detect Potential Low Birth Weight Cases	##########\n")
	print(f"HYPER PARAMETERS OF THE MODEL:\n")
	print(f"Neural Net Architecture:")
	print(f"\t Number of Nodes in Input layer : {nn.Ip}")
	print(f"\t Number of Nodes in Hidden layer : {nn.Hidden}")
	print(f"\t Number of Nodes in Hidden layer : {nn.Op}")
	print(f"Initial Learning Rate : {nn.lr}")
	print(f"Test - Train split : 90-10")
	print(f"Number of epochs for which the model was trained: {epochs} epochs\n")

	logging.info(f"##########	Nueral Network to detect Potential Low Birth Weight Cases	########## ")
	logging.info(f"HYPER PARAMETERS OF THE MODEL:\n")
	logging.info(f"Nueral Net Architecture:")
	logging.info(f"\t Number of Nodes in Input layer : {nn.Ip}")
	logging.info(f"\t Number of Nodes in Hidden layer : {nn.Hidden}")
	logging.info(f"\t Number of Nodes in Hidden layer : {nn.Op}")
	logging.info(f"Initial Learning Rate : {nn.lr}")
	logging.info(f"Test - Train split : 90-10\n")

	print("Training...")
	logging.info(f"Training...")
	startTime = datetime.now().microsecond
	yobs=nn.train(X_train,y_train,epochs)
	endTime=datetime.now().microsecond
	time=(endTime - startTime)/1000000.0
	print(f"Execution time in seconds = {time} seconds")
	logging.info(f"Execution time in seconds = {time}")
	
	loss1=np.mean(np.square(yobs-y_train))
	acc1=(1-loss1)*100
	print("Train loss:\t",loss1)
	logging.info(f"Train loss: {loss1}")
	print("Train accuracy:\t",(1-loss1)*100,"%")
	logging.info(f"Train accuracy:{acc1}%")
	print("\n")

	print("Testing...")
	logging.info(f"Testing...")
	y_test_obs=nn.predict(X_test)
	loss2=np.mean(np.square(y_test_obs-y_test))
	acc2=(1-(loss2))*100;
	logging.info(f"Test loss: {loss2}")
	print("Test loss:\t",loss2)
	logging.info(f"Test accuracy: {acc2}%")
	print("Test accuracy:\t",(1-loss2)*100,"%")


	# plotting a confusion matrix
	cm,p,r,f1 = CM(y_test, y_test_obs)
	print("\n")
	print("Confusion Matrix : ")
	print(cm)
	print("\n")
	print(f"Precision : {p}")
	print(f"Recall : {r}")
	print(f"F1 SCORE : {f1}")
	logging.info(f"\n")
	logging.info(f"Confusion Matrix : ")
	logging.info(f"{cm}")
	logging.info("\n")
	logging.info(f"Precision : {p}")
	logging.info(f"Recall : {r}")
	logging.info(f"F1 SCORE : {f1}")

	# print(nn.l)
	# print(nn.error)
	# #plotting
	# plt.plot(nn.error,nn.l)
	# plt.xlabel('Error rate') 
	# plt.ylabel('learning rate') 
	# plt.title('LR DECAY : Error rate Vs learning Rate ')
	# plt.savefig('lr_decay') 
  







