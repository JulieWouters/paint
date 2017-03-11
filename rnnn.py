#import pandas as pd
import sys
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, Masking
from keras.layers import SimpleRNN, LSTM
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


def pad_sequences_higher_dim(sequences, maxlen=None, dim=1, dtype='int32',
    padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
#    import pdb; pdb.set_trace()
    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            replace = trunc
            x[idx, :len(trunc)] = replace
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def getFeatureVectors(input, window):

    # ax = fig.add_subplot(131)
    # ax.set_title('imshow: equidistant')
    # im = plt.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # plt.show()

    allRows = [[float(j) for j in i] for i in input]
    distances = [(i[0]-allRows[0][0]) for i in allRows]
    xedges = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3, 2*math.pi]
    yedges = list(np.logspace(-1, 3.2, num=6))
    #yedges = [0, 50000]
    print(xedges)

    j = 0
    input_data = []
    sequence = []
    for row in allRows:
        if(all(map(lambda x: x==0.0, row))):
            j=j+1
            if(len(sequence)>0):
                sequence = sequence[1::2]
                input_data.append(np.array(sequence))
            sequence = []
            continue

        xdiff = allRows[j][0]-allRows[1][0]
        ydiff = allRows[j][1]-allRows[1][1]
        r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
        x_minus = min(1,math.sqrt(r/window))
        xdiff = allRows[j][0]-allRows[len(allRows)-2][0]
        ydiff = allRows[j][1]-allRows[len(allRows)-2][1]
        r = math.sqrt(xdiff*xdiff+ydiff*ydiff)
        x_plus = min(1,math.sqrt(r/window))
        feature_vector = [] #[x_minus, x_plus] klopt niet

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

        sequence.append(feature_vector)
        j=j+1

    x_data = input_data
    y_sequence = []
    y_input = []
    for row in allRows:
        if(all(map(lambda x: x==0.0,row))):
            if(len(y_sequence)>0):
                y_sequence = y_sequence[1::2]
                y_input.append(y_sequence)
                y_sequence = []
            continue
        y_sequence.append([row[3], row[4]])
    y_data = y_input

    # shape_context = np.array(x_data[2:])
    # max = np.amax(shape_context)
    # x_data[2:] = np.divide(x_data[2:],max)

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

#print(input_data)

# out = [item[4] for item in input_data]
# plt.scatter(np.array(range(0,len(out))), np.array(out),  color='black')
# plt.xticks(())
# plt.yticks(())
# plt.axis([0,len(out),-1,1])
# plt.show()

for window in [30,40]:
    for stroke in [30, 40, 50, 60, 70, 80, 90, 100]:
        for batch_size in [1, 5, 10, 20, 32, 64]:

            x_data, y_data = getFeatureVectors(input_data,window)

            # x_train = []
            # y_train = []
            # for i in range(0,len(x_data)):
            #     for j in range(0,len(x_data[i])):
            #         point_x = []
            #         point_y = []
            #         for k in range(24, 0, -1):
            #             if (j-k) >= 0:
            #                 point_x.append(x_data[i][j-k])
            #                 point_y.append(y_data[i][j-k])
            #             else:
            #                 point_x.append([0]*62)
            #                 point_y.append([0])
            #         point_x.append(x_data[i][j])
            #         point_y.append(y_data[i][j])
            #         x_train.append(point_x)
            #         y_train.append(point_y)


            x_train = []
            y_train = []
            for i in range(0,len(x_data)):
                for j in range(0,len(x_data[i]),stroke):
                    point_x = []
                    point_y = []
                    for k in range(0, stroke):
                        if (j+k) < len(x_data[i]):
                            point_x.append(x_data[i][j+k])
                            point_y.append(y_data[i][j+k])
                        else:
                            point_x.append([-1]*len(x_data[0][0]))
                            point_y.append([0,0])
                    x_train.append(point_x)
                    y_train.append(point_y)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            # print('x train:')
            # print(np.array(x_train).shape)
            # print(np.array(x_data).shape)
            # print(np.array(y_data).shape)

            # x_data = pad_sequences_higher_dim(x_data,dim=62)
            # print(x_data)

            #y = pad_sequences_higher_dim(y_data)
            #print(np.array(x_data).shape)
            # y_data = np.array(y_data)
            # y = np.array(y)
            # for i in range(0,len(y_data)):
            #     for j in range(0,len(y_data[i])):
            #         print(y[i,j])
            #         print(y_data[i][j])
            #         y[i,j][0] += y_data[i][j][0]

            # print(y)

            # y = []
            # lengths = [len(s) for s in y_data]
            # length = np.max(lengths)
            # rowofzeros = [0] * length
            # for i in range(0,len(y_data)):
            #     row = [[0]] * length
            #     for j in range(0,len(y_data[i])):
            #         row[j] = y_data[i][j]
            #     y.append(row)

            # print(y)
            # y_data = y

            # x_data = np.array(x_data)
            # y_data = np.array(y_data)

            # x_data = x_data[:,:50,:]
            # y_data = y_data[:,:50,:]

            # x_data = np.resize(x_data,(x_data.shape[0],20,x_data.shape[2]))
            # y_data = np.resize(y_data,(y_data.shape[0],20,y_data.shape[2]))

            # print(y_data)

            # x_test = [x_train[len(x_train)-1]]
            # x_data = x_data[0:len(x_data)-2]
            # y_test = [y_data[len(y_data)-1]]
            # y_data = y_data[0:len(y_data)-2]

            amount = round(len(x_train) / 50)

            x_test = x_train[len(x_train)-amount:]
            y_test = y_train[len(y_train)-amount:]

            x_train = x_train[:len(x_train)-amount]
            y_train = y_train[:len(y_train)-amount]

            # print(np.array(x_test).shape)

            # print(len(input.readlines()))

            # df = pd.read_csv(filename)
            # df.info()
            # df.ix[0:3,'y']

            # yourResult = [line.split(',') for line in input.readlines()]
            # print(yourResult)
            # for line in input.readlines():
            #   print(line)

            # model = Sequential()
            # model.add(SimpleRNN(32,input_shape=(x_data.shape[1:]),return_sequences=True))
            # model.add(TimeDistributed(Dense(1)))
            # #model.add(Dense(1))
            # model.add(Activation('sigmoid'))

            # rmsprop = RMSprop(lr=1e-6)
            # model.compile(loss='mean_squared_error', optimizer='sgd')
            #model.summary()
            # model.fit(np.asarray(x_data),np.asarray(y_data), nb_epoch=20, batch_size=500, verbose=1,
            #     validation_data=(np.array(x_test), np.array(y_test)))

            model = Sequential()
            # print(np.array(x_train).shape)
            # print(x_train.shape[1:])
            model.add(Masking(mask_value=-1,input_shape=(x_train.shape[1:])))
            model.add(LSTM(16, return_sequences=True))
            model.add(Activation('tanh'))
            model.add(TimeDistributed(Dense(2)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            for it in range(1,2):
                print('iteration: '+ str(it))
                model.fit(np.asarray(x_train),np.asarray(y_train), nb_epoch=150, batch_size=batch_size, verbose=2,
                    validation_data=(np.array(x_test), np.array(y_test)))

                scores = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
                print('RNN test score:', scores)

                out = model.predict(np.array(x_test),batch_size=batch_size, verbose=0)
                out = [x for y in out for x in y]
                predict = out
                out = [item[0] for item in out]
                y_testt = [x for y in y_test for x in y]
                y_test_1 = [item[0] for item in y_testt]
                out = np.array(out)
                predict = np.array(predict)
                y_testt = np.array(y_testt)
                y_test_1 = np.array(y_test_1)
                out = out.flatten()
                # print(out)

                #y_test = np.array(y_test).flatten()
                # out = [item[0] for item in out]
                out = np.array(out)

                # print(y_test)
                # print(out)
                # print(out.shape)

                # print(np.array(range(0,len(x_test))))
                # print(np.array(range(0,len(y_test))).shape)
                # print(np.array(y_test).shape)

                # print(np.array(range(0,len(y_test))).size)
                # print(np.array(out).size)
                fig = plt.figure(figsize=(20,20))
                plt.scatter(np.array(range(0,len(y_test_1))), np.array(y_test_1),  color='black')
                plt.plot(np.array(range(0,len(y_test_1))), np.array(out), color='blue', linewidth=3)

                mse = 0
                for i in range(0,len(y_testt)):
                    mse = mse + (predict[i]-y_testt[i])*(predict[i]-y_testt[i])
                mse = mse / len(y_testt)
                print('MSE: ' + str(mse))

                y_traint = [x for y in y_train for x in y]
                y_train_1 = [item[0] for item in y_traint]
                y_train_2 = [item[1] for item in y_traint]
                y_train_1 = np.array(y_train_1)
                y_train_2 = np.array(y_train_2)
                avg_1 = np.average(y_train_1)
                avg_2 = np.average(y_train_2)

                mse = 0
                for i in range(0,len(y_testt)):
                    mse = mse + (avg_1-y_testt[i])*(avg_1-y_testt[i])
                mse = mse / len(y_testt)
                print('MSE1: ' + str(mse))

                mse = 0
                for i in range(0,len(y_testt)):
                    mse = mse + (avg_2-y_testt[i])*(avg_2-y_testt[i])
                mse = mse / len(y_testt)
                print('MSE2: ' + str(mse))

                avg = np.average(y_traint)
                mse = 0
                for i in range(0,len(y_testt)):
                    mse = mse + (avg-y_testt[i])*(avg-y_testt[i])
                mse = mse / len(y_testt)
                print('MSE: ' + str(mse))

                # plt.xticks(())
                # plt.yticks(())
                plt.axis([0,len(y_testt),-1,1])
                plt.savefig(filename + ' ' + str(window)+' '+str(stroke)+' '+str(batch_size)+' '+str(it)+'.png')