import codecs
import numpy as np

with codecs.open('news2_evaluate.txt','r','utf-8') as file:
    labels = file.readline().strip().split()
    matrix = [[int(s) for s in  line.strip().split()] for line in file if len(line.strip())>0]
    # print(labels)
    # print(matrix)

    rate = np.zeros([len(labels),len(labels)])
    for i in range(len(labels)):
        s= sum(matrix[i])
        if s>0:
            for j in range(len(labels)):
                rate[i][j]=matrix[i][j]/s
    error = {}
    for i in range(len(labels)):
        for j in range(len(labels)):
            if rate[i][j]>0.05:
                error[labels[i],labels[j]]=rate[i][j]

    for item in error.items():
        print(item)