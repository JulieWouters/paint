import numpy as np
import math
import sys
import csv
import matplotlib.pyplot as plt

# print out names with speed color
def show_stroke(input):
    if(len(input)>0):
        xstart = input[0][0]
        ystart = -input[0][1]

        xcoord = []
        ycoord = []
        for row in input:
            x = row[0]-xstart
            y = -row[1]-ystart
            xcoord.append(x)
            ycoord.append(y)

        length = len(xcoord)
        color = np.arange(length)
        plt.scatter(xcoord, ycoord,  c=color)

def compare_tilt(input):
    index = 0
    new_input = []
    for row in input:
        new_input.append([index,row])
        index += 1
    #fig = plt.figure(figsize=(45,9))
    #plt.axis([-10,450,-50,40])
    for i in range(0,5):
        fig = plt.figure()
        plt.title('ytilt')
        total_xtilt = []
        total_ytilt = []
        for row in new_input:
            if(row[0]%5 == i):
                stroke = row[1]
                xstart = stroke[0][0]
                ystart = -stroke[0][1]

                xcoord = []
                ycoord = []
                xtilt = []
                ytilt = []
                for p in stroke:
                    x = p[0]-xstart
                    y = -p[1]-ystart
                    xcoord.append(x+70*i)
                    ycoord.append(y)
                    xtilt.append(p[3])
                    ytilt.append(p[4])

                total_xtilt = total_xtilt + xtilt
                total_ytilt = total_ytilt + ytilt
                length = len(xcoord)
                color = np.array(ytilt)
                plt.scatter(xcoord, ycoord, s=20, c=color, vmin=0.40, vmax = 0.90)
        print(' ')
        print('Letter ' + str(i))
        print('Mean xtilt: ')
        print(np.mean(np.array(total_xtilt)))
        print('Standard deviation xtilt: ')
        print(np.std(np.array(total_xtilt)))
        print('Mean ytilt:')
        print(np.mean(np.array(total_ytilt)))
        print('Standard deviation ytilt:')
        print(np.std(np.array(total_ytilt)))
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        plt.title('Drawing speed')
        for row in new_input:
            if(row[0]%5 == i):
                show_stroke(row[1])
        plt.show()
        plt.close()
    plt.savefig('fig.png')
    plt.show()
    plt.close()


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

allRows = [[float(j) for j in i] for i in input_data]
sequence = []
strokes = []
for row in allRows:
    if(all(map(lambda x: x==0.0, row))):
        # fig = plt.figure()
        # print([item[3] for item in sequence])
        # plt.scatter(np.arange(len(sequence)),[item[3] for item in sequence])
        # plt.show()
        # show_stroke(sequence)
        if(len(sequence) > 0):
            strokes.append(sequence)
        sequence = []
        continue
    sequence.append(row)

compare_tilt(strokes)