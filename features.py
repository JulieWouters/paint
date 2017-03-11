import math
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_length(point1,point2):
    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]
    length = math.sqrt(xdiff*xdiff + ydiff*ydiff)
    return (length)

def get_closure(input):
    #length of trajectory
    traj_length = 0
    l = 0
    for row in range(0,len(input)-1):
        l += 1
        length = get_length(input[row],input[row+1])
        traj_length = traj_length + length

    #distance between first and last point on stroke
    if(len(input)> 1):
        dist = get_length(input[0],input[len(input)-1])
        if(dist == 0):
            closure = 0
        else:
            closure = traj_length / dist
    else:
        closure = 0
    return closure

def get_directional_features(input):
    #initial angle (between first and fifth point because of less noise)
    if(len(input) > 5):
        length = get_length(input[0],input[4])
        if(length == 0):
            cos_init = 0
            sin_init = 0
        else:
            cos_init = (input[4][0] - input[0][0]) / length
            sin_init = (input[4][1] - input[0][1]) / length
    else:
        cos_init = math.cos(0)
        sin_init = math.sin(0)

    #angle between first and last point
    if(len(input)> 1):
        length = get_length(input[0],input[len(input)-1])
        if(length == 0):
            cos_fl = 0
            sin_fl = 0
        else:
            cos_fl = (input[len(input)-1][0] - input[0][0]) / length
            sin_fl = (input[len(input)-1][1] - input[0][1]) / length
    else:
        cos_fl = math.cos(0)
        sin_fl = math.sin(0)

    #average direction + curvature + average curvature
    av_dir = 0
    curv = 0
    av_curv = 0
    for row in range(0,len(input)-1):
        ydiff = input[row+1][1] - input[row][1]
        xdiff = input[row+1][0] - input[row][0]
        if(xdiff != 0):
            av_dir = av_dir + math.atan2(ydiff,xdiff)

        #for curvature
        if(row == 0):
            continue
        ydiff2 = input[row][1] - input[row-1][1]
        xdiff2 = input[row][0] - input[row-1][0]
        numerator = xdiff*xdiff2 + ydiff*ydiff2
        denominator = get_length(input[row],input[row-1])*get_length(input[row+1],input[row])
        if(denominator == 0):
            curv = curv + math.acos(0)
        else:
            angle = numerator/denominator
            if(angle > 1.0 or angle < -1.0):
                angle = round(angle)
            curv = curv + math.acos(angle)

    if(len(input)>1):
        av_dir = av_dir / (len(input) - 1)
    if(len(input)>2):
        av_curv = curv / (len(input) - 2)

    #stdev of curvature
    stdev_curv = 0
    for row in range(1,len(input)-1):
        ydiff = input[row+1][1] - input[row][1]
        xdiff = input[row+1][0] - input[row][0]

        ydiff2 = input[row][1] - input[row-1][1]
        xdiff2 = input[row][0] - input[row-1][0]
        numerator = xdiff*xdiff2 + ydiff*ydiff2
        denominator = get_length(input[row],input[row-1])*get_length(input[row+1],input[row])
        if(denominator == 0):
            stdev_curv = stdev_curv + math.pow((math.acos(0) - av_curv),2)
        else:
            angle = numerator/denominator
            if(angle > 1.0 or angle < -1.0):
                angle = round(angle)
            stdev_curv = stdev_curv + math.pow((math.acos(angle) - av_curv),2)
    if(len(input)>2):
        stdev_curv = math.sqrt(stdev_curv / (len(input)-2))

    return (cos_init, sin_init, cos_fl, sin_fl, av_dir, curv, av_curv, stdev_curv)

def get_centroidal_radius(input):
    xs = [item[0] for item in input]
    ys = [item[1] for item in input]
    xs = np.array(xs)
    ys = np.array(ys)
    xcentroid = np.mean(xs)
    ycentroid = np.mean(ys)
    av_cr = 0
    for row in range(0, len(input)-1):
        av_cr = av_cr  + get_length(input[row],[xcentroid,ycentroid])
    if(len(input)>0):
        av_cr = av_cr / len(input)

    stdev_cr = 0
    for row in range(0, len(input)-1):
        stdev_cr = stdev_cr + math.pow((get_length(input[row],[xcentroid,ycentroid]) - av_cr),2)
    if(len(input)>0):
        stdev_cr = math.sqrt(stdev_cr / len(input))

    return (av_cr, stdev_cr)

def getFeatureVectors(input, window, filename):


    allRows = [[float(j) for j in i] for i in input]
    j = 0
    input_data = []
    for row in allRows:
        if(all(map(lambda x: x==0.0, row))):
            j=j+1
            continue

        feature_vector = []

        # Future feature vector
        x = []
        y = []
        xcoord = []
        ycoord = []
        for i in range(1, window+1):
            if(all(map(lambda x: x==0.0,allRows[j+i]))):
                break
            xf = allRows[j+i][0]
            yf = allRows[j+i][1]
            x.append(allRows[j+i])
            xdiff = xf-allRows[j][0]
            ydiff = yf-allRows[j][1]
            xcoord.append(xf)
            ycoord.append(yf)

        features = list(get_directional_features(x)) + list(get_centroidal_radius(x)) + [get_closure(x)]
        features = [round(item,2) for item in features]
        feature_vector.append(features)

        if(j%20 == 1 and j <= 1000):
            fig = plt.figure()
            plt.title(str(features))
            plt.scatter(xcoord, ycoord,  color='black')
            plt.scatter([allRows[j][0]],allRows[j][1], color='red')
            plt.axis([-700,700,-400,400])
            plt.savefig(filename + ' ' + str(j)+' future stroke.png')
            plt.close()

        # History histogram 
        x=[]
        y=[]
        xcoord = []
        ycoord = []
        for i in range(1,window+1):
            if(all(map(lambda x: x==0.0,allRows[j-i]))):
                break
            # if(i%4 > 0):
            #     continue
            xh = allRows[j-i][0]
            yh = allRows[j-i][1]
            x.append(allRows[j-i])
            xdiff = xh-allRows[j][0]
            ydiff = yh-allRows[j][1]
            xcoord.append(xh)
            ycoord.append(yh)

        features = list(get_directional_features(x)) + list(get_centroidal_radius(x)) + [get_closure(x)]
        features = [round(item,2) for item in features]
        feature_vector.append(features)

        if(j%20 == 1 and j <= 1000):
            fig = plt.figure()
            plt.title(str(features))
            plt.scatter(xcoord, ycoord,  color='black')
            plt.scatter([allRows[j][0]],allRows[j][1], color='red')
            plt.axis([-700,700,-400,400])
            plt.savefig(filename + ' ' + str(j)+' history stroke.png')
            plt.close()

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

getFeatureVectors(input_data,40,filename)

allRows = [[float(j) for j in i] for i in input_data]
sequence = []
for row in allRows:
    if(all(map(lambda x: x==0.0, row))):
        if(len(sequence)>0):
            get_closure(sequence)
        sequence = []
        continue
    sequence.append(row)