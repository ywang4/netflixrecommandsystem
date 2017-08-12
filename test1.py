from numpy import *
from numpy import linalg as la
from collections import OrderedDict
from itertools import repeat
from math import *
import sys
import os.path
import os
import operator

def mytest():
	data = loadtxt("train.txt") #load train.txt as array
	with open('truth10.txt','w') as f:
		for user in range(150,200):
			print(user)
			count = 0
			for movie in range(1000):
			# 	f.write(str(int(data[user][movie])) + ' ')
			# f.write('\r\n')
				if (data[user][movie] != 0):
					if(count < 10):
						# f.write(str(user) + ' ' + 
					 #    		str(movie+1) + ' ' + 
					 #    		str(int(data[user][movie])))
						# f.write('\r\n') #on Windows
						count += 1
					else:
						f.write(str(user) + ' ' + 
						   		str(movie+1) + ' ' + 
						   		str(int(data[user][movie])))
								#'0')
						f.write('\r\n') #on Windows

def result():
	truth10 = open('truth10.txt','r')
	data = []

	for line in truth10:
		#seperate numbers by space
		user, movie, rating = line.split()
		
		data.append(float(rating))


	result10 = open('myresult.txt','r')
	data2 = []

	for line in result10:
		#seperate numbers by space
		user, movie, rating = line.split()
		
		data2.append(float(rating))

	truth10.close()
	result10.close()
	
	sum = 0.0
	for i in range(size(data)):
		
		sum += (absolute(data2[i] - data[i]))

	a = float(size(data))
	sum /= a
	return sum



