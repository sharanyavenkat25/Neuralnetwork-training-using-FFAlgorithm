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
from Neural_Network import NN
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
np.random.seed(5)
nn=NN(1,3,1)
curr_weights=nn.weightsIH
epochs= 15
data = pd.read_csv('train.csv')
X=data[['x']]
y=data[['y']]
def weights_to_vector(nn):
	'''
	converts all the weight matrices in a nn to a vector
	'''

	mat1=np.reshape(nn.weightsIH,-1)
	mat2=np.reshape(nn.weightsHO,-1)
	index=len(mat1) # remember where we concatenated
	mat = np.concatenate((mat1,mat2))
	return mat

def vector_to_weights(nn,new_weights):
	'''
	converts all vectors outputed by FF back to weight matrices
	'''
	index=3 ## for this case
	mat1=new_weights[0:index]
	mat2=new_weights[index:]
	nn.weightsIH=np.reshape(mat1,np.shape(nn.weightsIH))
	nn.weightsHO=np.reshape(mat2,np.shape(nn.weightsHO))
def fitness_function(nn,weights):
	vector_to_weights(nn,weights)
	yhat=nn.predict(X)
	# print("Expected Values\n",y)
	# print("initial Predicted values...\n",yhat)
	error=abs(np.mean(np.square(y-yhat)))
	# print("error:\n",error)
	return error

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
def ff():
	#initialise parameters
	n=20 #Total Number of fireflies
	dim=6 #Dimension of each firefly
	x=np.random.rand(20,6) # Entire population of fireflies
	error=np.zeros(n)
	beta = 1 #always 1
	gamma = 1 #absorption coeff
	# alpha = random.uniform(0,1) #randomization parameter
	alpha = 0.8
	print("Initial Population\n",x)

	#finding the value objective function for each firefly
	for i in range(n):
		error[i]=fitness_function(nn,x[i])
	print("Fitness Function value for each firefly is : \n",error)
	li=[0]*n #light intensity of each firefly li[i]=1/f(xi)

	for i in range(len(error)):
		li[i]=1/error[i]
	print("Light Intensity of population..\n",li)

	# Training
	t=0
	while(t < epochs):
		for i in range(0,n):
			for j in range(0,i):
				if(li[j]>li[i]):
					rand=random.uniform(0,1)
					r=distance.euclidean(x[i],x[j])#cartesian distance
					x[i]=x[i]+beta*math.exp(-gamma * math.pow(r, 2.0))*(x[i]-x[j])+alpha*(rand-0.5) #updating x[i]'s

		#update li
		for i in range(n):
			error[i]=fitness_function(nn,x[i])
		li=[0]*n #light intensity of each firefly li[i]=1/f(xi)

		for i in range(len(error)):
			li[i]=1/error[i]
		t=t+1

	#Sort fireflies based on li and pick best (highest li)
	print("Finished Training")
	sort_ff(x,li,n,dim)
	best = x[0]
	#put it back in the nn
	vector_to_weights(nn,best)
	yhat=nn.predict(X)
	print("Expected Values\n",y)
	print("Predicted values...\n",yhat)
	error=abs(np.mean(np.square(y-yhat)))





	


ff()




