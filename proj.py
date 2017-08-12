from numpy import *
from numpy import linalg as la
from collections import OrderedDict
from itertools import repeat
from math import *
import sys
import os.path
import os
import operator

#using Euclidian distance to calculate the similarity
def ecludSim (matA, matB):
	return 1.0/(1.0 + la.norm(matA - matB))

#Pearson correlation
#compute the difference between vector and its avergae
#calculate the cosine sim between difference of two array
def pearsSim (inA, inB):
	A = inA
	B = inB
	meanA = mean(A)
	meanB = mean(B)

	for i in range(len(A)):
		A[i] = A[i] - meanA


	for j in range(len(B)):
		B[j] = B[j] - meanB

	r = dot(A,B)
	result = la.norm(A) * la.norm(B)
	if(result != 0.0):
		result = r / result
	else:
		result = 0.0

	return result

#using cosine similarity
def cosSim(matA, matB):
	r = float(matA * matB.T) #inner product between two vector
	nom = la.norm(matA)*la.norm(matB) #product of lenght of two vector
	if(nom == 0.0):
		return 0.0
	else:
		return 0.5+0.5*(r/nom)
		#return r/nom

#load trainingData in a matrix with row = user and col = moive rating
def trainingData():
	data = loadtxt("train.txt")
	#data = loadtxt("mytrain.txt")
	return mat(data)

#IUF
def iuf(matA):
	iuf = [0] * 1000
	for row in matA:
		row = row.tolist()
		row = row[0]
		m = 0
		for i in range(1000):
			if(row[i] != 0):
				iuf[i] +=1

	for m in range(len(iuf)):
		if(iuf[m] != 0):
			iuf[m] = log10(200/iuf[m])
		else:
			iuf[m] = 1

	x = 0
	for row in matA:
		row = row.tolist()
		row = row[0]
		for i in range(len(row)):
			matA[x,i] = row[i]*iuf[i]
		x+=1


#case ampification
def caseamp(srate):
	for i in range(len(srate)):
		srate[i] = srate[i]**2.5
		if (isnan(srate[i])):
			srate[i] = 0.0


#load test data in dictionary in formart of 
#    userid : [0]corresponding rating in matrix, 
#			  [1]array of rated movie id, 
#             [2]array of unrated movie id
#			  [3]predict rating of unrated movie
def testData( filename ):
	#open test5 or test10 or test20
	f = open(filename,'r')
	#create a dictionay and put user id as keys
	data = {}
	for line in f:
		#seperate numbers by space
		user, movie, rating = line.split()
		#turn string to int
		user = int(user)
		#put a list of 3 lists in value
		data[user] = [[],[],[],[]]

	f.close()

	#reopen the file and start to load data
	f = open(filename,'r')
	#store data of rating and movie id in corresponding lists
	for line in f:
		#seperate numbers by space
		user, movie, rating = line.split()
		#turn string to int
		user = int(user)
		movie = int(movie)
		rating = float(rating)

		if(rating != 0.0):
			data[user][0].append(rating)
			data[user][1].append(movie)
		else:
			data[user][2].append(movie)
			data[user][3].append(0.0)

	#close the file and sort dictionary in order
	f.close()
	data = OrderedDict(sorted(data.items()))
	return data

def adjustedCos(unmovie, movie,train):
	#load training data as array
	#train = loadtxt("mytrain.txt")
	#train = loadtxt("train.txt")
	# sum = 0.0
	# sqri = 0.0
	# sqrj = 0.0
	ratei = []
	ratej = []
	for user in range(shape(train)[0]):
		#find the mean rating of this user
		meanuser = 0.0
		countuser = 0.0
		for rate in train[user]:
			if (rate != 0.0):
				meanuser += rate
				countuser += 1.0
		meanuser /= countuser

		

		#if both of movies are not rated 0
		#calculate i -mean, j-mean, (i-mean)**2, (j-mean)**2
		if((train[user][unmovie-1] * train[user][movie-1] )!= 0.0 ):
			# sum += ((train[user][unmovie] - meanuser) * 
			# 	    (train[user][movie] - meanuser))
			# sqri += ((train[user][unmovie] - meanuser)**2)
			# sqrj += ((train[user][movie] - meanuser)**2)
			ratei.append((train[user][unmovie-1] - meanuser))
			ratej.append((train[user][movie-1] - meanuser))

	ratei = array(ratei)
	ratej = array(ratej)
	r = dot(ratei,ratej) #inner product between two vector
	nom = la.norm(ratei)*la.norm(ratej) #product of lenght of two vector
	if(nom == 0.0):
		return 0.0
	else:
		return (0.5+0.5*r/nom)





		



#return list of similar user with userID
#only use cosine similarity or eclud similarity
#should take most k similar users and then calculate the
#predicted rate for the movie
def predict1(filename, resultfile, count):
	#load data from test file and train file
	test = testData(filename)
	train = trainingData()
	#do IUF to training data
	#iuf(train)

	for user in test.keys():
		#trainrate = [[] for i in repeat(None, 200)]
		trainrate = [[] for i in repeat(None, 150)]
		sim = []
		for row in train: #row = user
			row = row.tolist()
			#put two ratings in matrix
			rating = []
			for c in range(count):
				rating.append(row[0][test[user][1][c]])
			rating = mat(rating)
			rating2 = mat(test[user][0])

 			#calculating the similarity
			#using ecluSim
			#This is my own algorithm to calculate the similarity 
			#sim.append(ecludSim(rating[0,:],rating2[0,:]))

			#using cosineSim
			sim.append(cosSim(rating[0,:],rating2[0,:]))

		#movieid range: 0-999
		#save the rate of corresponding movie from train rate
		trainuser = 0;
		mattrain = trainingData()
		for row in mattrain:
			row = row.tolist()
			for movieid in test[user][2]:
					trainrate[trainuser].append(row[0][movieid-1])
			trainuser += 1 #move to next row

			

		#could sort the training data based on sim
		#then take the top k users
		#calculating predict rate for userid = user
		#sortedrate is turned into list where sortedrate[n][0] is user id
		#									  sortedrate[n][1] is sim
		unsortedrate = {}
		k = 200
		for userid in range(len(sim)):
			unsortedrate[userid+1] = sim[userid]
		sortedrate = sorted(unsortedrate.items(), key=operator.itemgetter(1),
							reverse=True)[:k]
		srate = {}
		#turn sortedrate into dictionary called srate
		for i in range(len(sortedrate)):
			srate[sortedrate[i][0]] = sortedrate[i][1]

		#print(srate)
		#caseamp(srate)

		for i in range(len(test[user][2])): #movie id
			sumSim = 0.0
			for j in srate.keys(): #user id
				if(trainrate[j-1][i] != 0):
					test[user][3][i] += trainrate[j-1][i] * srate[j]
					sumSim += srate[j]
			#having problem in these codes
			#print(sumSim)
			if(sumSim != 0.0):
				test[user][3][i] /= sumSim
				test[user][3][i] = int(round(test[user][3][i]))
				if(test[user][3][i] == 0):
					test[user][3][i] = 1
			if(test[user][3][i] == 0):
				test[user][3][i] = 3
	#write result in *.txt
	with open(resultfile,'w') as f:
		for user in test.keys():
			for i in range(len(test[user][2])):
				f.write(str(user) + ' ' + 
					    str(test[user][2][i]) + ' ' + 
					    str(test[user][3][i]))
				f.write('\r\n') #on Windows
				#f.write('\n') #on Mac

#use pearson correlation to calculate cosine similarity
def predict2(filename, resultfile, count):
	#load data from test file and train file
	test = testData(filename)
	train = trainingData()
	#iuf(train)
	
	for user in test.keys():
		trainrate = [[] for i in repeat(None, 200)]
		trainuser = 0;
		sim = []
		mean1 = {}
		for row in train: #row = user
			row = row.tolist()
			#put two ratings in matrix
			rating = []
			for c in range(count):
				rating.append(row[0][test[user][1][c]])
			rating = array(rating)
			rating2 = array(test[user][0])
			#print(rating)

			#calculating the mean rate of the whole row
			sum1 = 0.0
			count1 = 0.0
			for rate1 in row[0]:
				if (rate1 != 0.0):
					sum1 += rate1
					count1 += 1.0

			meanrate1 = sum1/count1

			#mean should be the mean of whole row
			mean1[trainuser] = meanrate1
	
 			#calculating the similarity
			sim.append(pearsSim(rating,rating2))
			#calculating the correlated avg of each training user
			trainuser+=1

			


		#print(mean1)
		#movieid range: 0-999
		#save the rate of corresponding movie from train rate
		trainuser = 0;
		mattrain = trainingData()
		for row in mattrain:
			row = row.tolist()
			for movieid in test[user][2]:
					trainrate[trainuser].append(row[0][movieid-1])
			trainuser += 1 #move to next row


			

		#could sort the training data based on sim
		#then take the top k users
		#calculating predict rate for userid = user
		#sortedrate is turned into list where sortedrate[n][0] is user id
		#									  sortedrate[n][1] is sim
		#wanted to rate the rating by absolute value and return a list of user id
		#take the top k users
		unsortedrate = {}
		k = 200
		for userid in range(len(sim)):
			unsortedrate[userid+1] = absolute(sim[userid])

		#unsortedrate is a dictionary with userid(1-200) and rating
		sortedrate = sorted(unsortedrate.items(), key=operator.itemgetter(1),
							reverse=True)[:k]
		#srate = {}
		userlist = []
		#turn sortedrate into dictionary called srate
		for i in range(50):
			userlist.append(sortedrate[i][0])
		#sortedrate[i][1]
		#print(userlist)
		#caseamp(sim)
		#dayu = 0

		for i in range(len(test[user][2])): #movie id
			sumSim = 0.0
			for j in userlist: #user id
				if(trainrate[j-1][i] != 0):
					#print(j-1)
					#print(sim[j-1])
					test[user][3][i] += ((trainrate[j-1][i] - mean1[j-1]) * sim[j-1])
					sumSim += absolute(sim[j-1]) 
			if(sumSim != 0.0):
				test[user][3][i] /= sumSim
			else:
				test[user][3][i] = 0.0
			rr = array(test[user][0])
			test[user][3][i] += mean(rr)
			test[user][3][i] = int(round(test[user][3][i]))
			if(test[user][3][i] == 0.0):
				test[user][3][i] = 1
			if(test[user][3][i] < 0):
				test[user][3][i] = 1

			if(test[user][3][i] >= 5):
				test[user][3][i] = 5
				#dayu+=1

		#print(dayu)

	#write result in *.txt
	with open(resultfile,'w') as f:
		for user in test.keys():
			for i in range(len(test[user][2])):
				f.write(str(user) + ' ' + 
					    str(test[user][2][i]) + ' ' + 
					    str(test[user][3][i]))
				f.write('\r\n') #on Windows
				#f.write('\n') #on Mac

#use item-based method and adjusted cosine similarity to predict
def predict3(filename, resultfile, count):
	test = testData(filename)
	#train = loadtxt("mytrain.txt")
	train = loadtxt("train.txt")
	#user is each userid in test data
	x = 0
	for user in test.keys():
		x+=1
		#unmovie is each unrated movie id in test data
		for i in range(len(test[user][2])):
			unmovie = test[user][2][i]
			sim = {}
			rate = 0.0
			#calculate sim between each unrated movie and rated movie
			for movie in test[user][1]:
				sim[movie] = adjustedCos(unmovie, movie,train)

			#print(sim)
			sumsim = 0.0
			for j in range(len(test[user][1])):
				rate += test[user][0][j] * sim[test[user][1][j]]
				sumsim += absolute(sim[test[user][1][j]])

			#print(str(sumsim) + ' ' + str(rate))
			if(sumsim == 0.0):
				rate = 3
			else:
				rate /= sumsim
				rate  = int(round(rate))
			if ( rate < 0):
				rate = 3
			if(rate == 0):
				rate = 1
			if( rate >= 5):
				rate = 5

			test[user][3][i] = rate
		print(x)

	#write result in *.txt
	with open(resultfile,'w') as f:
		for user in test.keys():
			for i in range(len(test[user][2])):
				f.write(str(user) + ' ' + 
					    str(test[user][2][i]) + ' ' + 
					    str(test[user][3][i]))
				f.write('\r\n') #on Windows






if __name__ == '__main__':
	if len(sys.argv) >2 :
			filename = str(sys.argv[1])
			resultfile = str(sys.argv[2])
			count = int(sys.argv[3])
			predict3(filename, resultfile, count)
	else:
		print('Usage: python3 proj.py filename resultfile countofTrainedrate')