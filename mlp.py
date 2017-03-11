#import pandas as pd
import sys
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

def getFeatureVectors(input, window):

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
        feature_vector = [x_minus, x_plus] #, allRows[j][2]]

        # Future histogram
        x = []
        y = []
        for i in range(1,window+1):
            if(all(map(lambda x: x==0.0,allRows[j+i]))):
                break
            xdiff = allRows[j+i][0]-allRows[j][0]
            ydiff = allRows[j+i][1]-allRows[j][1]
            r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
            y.append(r)
            if(r>0.0):
                theta = math.acos(math.fabs(xdiff)/r)
            else:
                theta = 0
            x.append(theta)

        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        feature_vector = feature_vector + [y for x in H for y in x]

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
    #   plt.show()

        # History histogram 
        x=[]
        y=[]
        for i in range(1,window+1):
            if(all(map(lambda x: x==0.0,allRows[j-i]))):
                break
            xdiff = allRows[j-i][0]-allRows[j][0]
            ydiff = allRows[j-i][1]-allRows[j][1]
            r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
            y.append(r)
            if(r > 0.0):
                theta = math.acos(math.fabs(xdiff)/r)
            else:
                theta = 0
            x.append(theta)

        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        feature_vector = feature_vector + [y for x in H for y in x]

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
    #   plt.show()

        input_data.append(feature_vector)
        j=j+1

    x_data = np.array(input_data)
    y_input = []
    for row in allRows:
        if(all(map(lambda x: x==0.0,row))):
                continue
        y_input.append([row[3],row[4]])
    y_data = np.array(y_input)

    # shape_context = np.array(x_data[2:])
    # max = np.amax(shape_context)
    # x_data[2:] = np.divide(x_data[2:],max)

    for i in range(0,len(x_data)):
        shape_context = np.array(x_data[i,2:])
        max = np.amax(shape_context)
        print(max)
        x_data[i,2:] = np.divide(x_data[i,2:],max)
    # print(x_data)

    x_data = x_data[1::2]
    #x_data = x_data[1::2]

    y_data = y_data[1::2]
    # y_data = y_data[1::2]

    return x_data,y_data


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

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

print(len(input_data))

x_data, y_data = getFeatureVectors(input_data,25)
x_data = np.array(x_data)
y_data = np.array(y_data)

# print(len(input.readlines()))

# df = pd.read_csv(filename)
# df.info()
# df.ix[0:3,'y']

# yourResult = [line.split(',') for line in input.readlines()]
# print(yourResult)
# for line in input.readlines():
#   print(line)

# nn = MLPRegressor(
#   hidden_layer_sizes=(32,), 
#   activation='tanh',
#   verbose=True)
# nn.fit(x_data,y_data)
# print(x_data[1,:].size)

# plt.scatter(np.array(range(0,len(x_data))), y_data,  color='black')
# plt.plot(np.array(range(0,len(x_data))), nn.predict(x_data), color='blue',
#          linewidth=3)

# # plt.scatter(np.array(range(0,20)), y_data[len(y_data)-21:len(y_data)-1],  color='black')
# # plt.plot(np.array(range(0,20)), nn.predict(x_data[len(x_data)-21:len(x_data)-1,:]), color='blue',
# #          linewidth=3)

batch = 10

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import SimpleRNN

model = Sequential()
model.add(Dense(32, input_dim=x_data.shape[1]))
model.add(Activation('tanh'))
# model.add(Dense(16))
# model.add(Activation('tanh'))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')

amount = round(len(x_data) / 10)

x_train = x_data[:len(x_data)-amount]
print(len(x_train))
y_train = y_data[:len(x_data)-amount]

x_test = x_data[len(x_data)-amount:]
y_test = y_data[len(x_data)-amount:]

model.fit(x_train, y_train, nb_epoch=100, batch_size=batch)
score = model.evaluate(x_test, y_test, batch_size=batch)

y_test_1 = [item[0] for item in y_test]
y_test_2 = [item[1] for item in y_test]

plt.scatter(np.array(range(0,len(x_test))), y_test_1,  color='black')

y_predict = model.predict(x_test, batch_size=batch, verbose=1)

y_predict_1 = [item[0] for item in y_predict]
y_predict_2 = [item[1] for item in y_predict]

plt.plot(np.array(range(0,len(x_test))), y_predict_1, color='blue',
         linewidth=3)

# plt.scatter(np.array(range(0,len(x_test))), y_test_2,  color='black')

# plt.plot(np.array(range(0,len(x_test))), y_predict_2, color='blue',
#          linewidth=3)

# plt.scatter(np.array(range(0,20)), y_data[len(y_data)-21:len(y_data)-1],  color='black')
# plt.plot(np.array(range(0,20)), svr.predict(x_data[len(x_data)-21:len(x_data)-1,:]), color='blue',
#          linewidth=3)

mse = 0
for i in range(0,len(y_test)):
    mse = mse + (y_predict[i]-y_test[i])*(y_predict[i]-y_test[i])
mse = mse / len(y_test)
print('MSE: ' + str(mse))

y_predict = model.predict(x_train, batch_size=batch, verbose=1)

mse = 0
for i in range(0,len(y_test)):
    mse = mse + (y_predict[i]-y_test[i])*(y_predict[i]-y_test[i])
mse = mse / len(y_test)
print('MSE: ' + str(mse))

y_train_1 = [item[0] for item in y_train]
y_train_2 = [item[1] for item in y_train]
avg_1 = np.average(y_train_1)
avg_2 = np.average(y_train_2)

mse = 0
for i in range(0,len(y_test)):
    mse = mse + (avg_1-y_test[i])*(avg_1-y_test[i])
mse = mse / len(y_test)
print('MSE1: ' + str(mse))

mse = 0
for i in range(0,len(y_test)):
    mse = mse + (avg_2-y_test[i])*(avg_2-y_test[i])
mse = mse / len(y_test)
print('MSE2: ' + str(mse))

plt.xticks(())
plt.yticks(())
plt.axis([0,len(x_test),-1,1])
plt.show()

print(x_data)