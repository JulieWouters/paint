import numpy as np
import math
import sys
import csv
import matplotlib.pyplot as plt

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
        fig = plt.figure()
        plt.scatter(xcoord, ycoord,  c=color)
        plt.show()

# print out names with speed color
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
for row in allRows:
    if(all(map(lambda x: x==0.0, row))):
        fig = plt.figure()
        print([item[3] for item in sequence])
        plt.scatter(np.arange(len(sequence)),[item[3] for item in sequence])
        plt.show()
        show_stroke(sequence)
        sequence = []
        continue
    sequence.append(row)