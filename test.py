#import pandas as pd
import sys
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as cm

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

def getFeatureVectors(input, window, filename):

	# ax = fig.add_subplot(131)
	# ax.set_title('imshow: equidistant')
	# im = plt.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

	# plt.show()

	allRows = [[float(j) for j in i] for i in input]
	distances = [(i[0]-allRows[0][0]) for i in allRows]
	xedges = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3, 2*math.pi]
	yedges = list(np.logspace(-1, 3.2, num=6))
	print(xedges)

	j = 0
	input_data = []
	for row in allRows:
		if(all(map(lambda x: x==0.0, row))):
			j=j+1
			continue

		xdiff = allRows[j][0]-allRows[1][0]
		ydiff = allRows[j][1]-allRows[1][1]
		r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
		x_minus = min(1,math.sqrt(r/window))
		xdiff = allRows[j][0]-allRows[len(allRows)-2][0]
		ydiff = allRows[j][1]-allRows[len(allRows)-2][1]
		r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
		x_plus = min(1,math.sqrt(r/window))
		feature_vector = [x_minus, x_plus]

		# Future histogram
		x = []
		y = []
		xcoord = []
		ycoord = []
		for i in range(1,4*window+1):
			if(all(map(lambda x: x==0.0,allRows[j+i]))):
				break
			# if(i%4 > 0):
			# 	continue
			xf = allRows[j+i][0]
			yf = allRows[j+i][1]
			xdiff = xf-allRows[j][0]
			ydiff = yf-allRows[j][1]
			xcoord.append(xf)
			ycoord.append(yf)
			r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
			y.append(r)
			if(r>0.0):
				theta = math.acos(math.fabs(xdiff)/r)
				if(xdiff<0.0 and ydiff<0.0):
					theta = math.pi + theta
				elif(ydiff<0.0 and xdiff>=0.0):
					theta = -theta + 2*math.pi
				elif(xdiff<0.0 and ydiff>=0.0):
					theta = math.pi - theta 
			else:
				theta = 0
			x.append(theta)

		print(xcoord)
		H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
		feature_vector = feature_vector + [y for x in H for y in x]

		if(j%20 == 1 and j <= 1000):
			fig = plt.figure()
			plt.scatter(xcoord, ycoord,  color='black')
			plt.scatter([allRows[j][0]],allRows[j][1], color='red')
			plt.axis([-700,700,-400,400])
			plt.savefig(filename + ' ' + str(j)+' future stroke.png')

			fig = plt.figure(figsize=(7, 4))
			ax = fig.add_subplot(132)
			ax.set_title('pcolormesh: exact bin edges')
			X, Y = np.meshgrid(xedges, yedges)
			ar = ax.pcolor(X, Y, np.transpose(H),cmap=cm.gray_r)
			plt.set_cmap(cm.gray_r)
			feature_vector = feature_vector + list(ar.get_array())
			im = plt.imshow(H, aspect=0.01, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
			# ax.set_aspect('equal')
			plt.colorbar()
			plt.yscale('log',nonposy='clip')
			plt.savefig(filename + ' ' + str(j)+' future hist.png')

		# History histogram 
		x=[]
		y=[]
		xcoord = []
		ycoord = []
		for i in range(1,4*window+1):
			if(all(map(lambda x: x==0.0,allRows[j-i]))):
				break
			if(i%4 > 0):
				continue
			xh = allRows[j-i][0]
			yh = allRows[j-i][1]
			xdiff = xh-allRows[j][0]
			ydiff = yh-allRows[j][1]
			xcoord.append(xh)
			ycoord.append(yh)
			r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
			y.append(r)
			if(r > 0.0):
				theta = math.acos(math.fabs(xdiff)/r)
				if(xdiff<0.0 and ydiff<0.0):
					theta = math.pi + theta
				elif(ydiff<0.0 and xdiff>=0.0):
					theta = -theta + 2*math.pi
				elif(xdiff<0.0 and ydiff>=0.0):
					theta = math.pi - theta 
			else:
				theta = 0
			x.append(theta)

		H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
		feature_vector = feature_vector + [y for x in H for y in x]

		if(j%20 == 1 and j <= 1000):
			fig = plt.figure()
			plt.scatter(xcoord, ycoord,  color='black')
			plt.scatter([allRows[j][0]],allRows[j][1], color='red')
			plt.axis([-700,700,-400,400])
			plt.savefig(filename + ' ' + str(j)+' history stroke.png')

			fig = plt.figure(figsize=(7, 4))
			ax = fig.add_subplot(132)
			ax.set_title('pcolormesh: exact bin edges')
			X, Y = np.meshgrid(xedges, yedges)
			ar = ax.pcolor(X, Y, np.transpose(H),cmap=cm.gray_r)
			feature_vector = feature_vector + list(ar.get_array())
			im = plt.imshow(H, aspect=0.01, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
			# ax.set_aspect('equal')
			plt.colorbar(cmap=cm.gray_r)
			plt.yscale('log',nonposy='clip')
			plt.savefig(filename + ' ' + str(j)+' history hist.png')

		# fig = plt.figure(figsize=(7, 7))
		# ax = fig.add_subplot(132)
		# ax.set_title('pcolormesh: exact bin edges')
		# X, Y = np.meshgrid(xedges, yedges)
		# ar = ax.pcolormesh(X, Y, np.transpose(H))
		# feature_vector = feature_vector + list(ar.get_array())
		# im = plt.imshow(H, aspect=0.01, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
		# # ax.set_aspect('equal')
		# plt.colorbar()
		# plt.yscale('log',nonposy='clip')
	#	plt.show()

		input_data.append(feature_vector)
		j=j+1

	x_data = np.array(input_data)
	y_input = []
	for row in allRows:
		if(all(map(lambda x: x==0.0,row))):
				continue
		y_input.append(row[4])
	y_data = np.array(y_input)
	return x_data,y_data


# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))

filename = sys.argv[1]
query = open(filename,'r')

input_data = []
for i in range(1,len(sys.argv)):
	filename = sys.argv[i]
	input = open(filename,'r')
	reader = csv.reader(input)
	allRows = [row for row in reader]
	# print(allRows)
	columns = allRows.pop(0)
	input_data = input_data + allRows

#print(input_data)

x_data, y_data = getFeatureVectors(input_data,40,filename)
stop
x_data = x_data[0:1800]
y_data = y_data[0:1800]

x_data = np.append(x_data,x_data,axis=0)
x_data = np.append(x_data,x_data,axis=0)
x_data = np.append(x_data,x_data,axis=0)

y_data = np.append(y_data,y_data,axis=0)
y_data = np.append(y_data,y_data,axis=0)
y_data = np.append(y_data,y_data,axis=0)

# print(len(input.readlines()))

# df = pd.read_csv(filename)
# df.info()
# df.ix[0:3,'y']

# yourResult = [line.split(',') for line in input.readlines()]
# print(yourResult)
# for line in input.readlines():
# 	print(line)

# nn = MLPRegressor(
# 	hidden_layer_sizes=(32,), 
# 	activation='tanh',
# 	verbose=True)
# nn.fit(x_data,y_data)
# print(x_data[1,:].size)

# x_predict = nn.predict(x_data)
# plt.scatter(np.array(range(0,len(x_data))), y_data,  color='black')
# plt.plot(np.array(range(0,len(x_data))), x_predict, color='blue',
#          linewidth=3)

# plt.scatter(np.array(range(0,20)), y_data[len(y_data)-21:len(y_data)-1],  color='black')
# plt.plot(np.array(range(0,20)), nn.predict(x_data[len(x_data)-21:len(x_data)-1,:]), color='blue',
#          linewidth=3)


svr = SVR(
	kernel='poly',
	verbose=True)
svr.fit(x_data,y_data)

plt.scatter(np.array(range(0,len(x_data))), y_data,  color='black')

x_predict = svr.predict(x_data)
plt.plot(np.array(range(0,len(x_data))), x_predict, color='blue',
         linewidth=3)

plt.scatter(np.array(range(0,20)), y_data[len(y_data)-21:len(y_data)-1],  color='black')
plt.scatter(np.array(range(0,20)), svr.predict(x_data[len(x_data)-21:len(x_data)-1,:]), color='blue')

mse = 0
for i in range(0,len(y_data)):
	mse = mse + (x_predict[i]-y_data[i])*(x_predict[i]-y_data[i])
mse = mse / len(y_data)
print('MSE: ' + str(mse))

plt.xticks(())
plt.yticks([-1,-0.5,0,0.5,1])
plt.axis([0,10000,-1,1])
plt.show()

print(x_data)