# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:04:37 2020

@author: Yus
"""

# -*- coding: utf-8 -*-


from pyspark import SparkContext, SparkConf
from math import sqrt
import time
import numpy
from pyspark.sql import SQLContext
from pyspark import sql
import random
import os


def computeDistance(x,y):
    return sqrt(sum([(a - b)**2 for a,b in zip(x,y)]))


def closestCluster(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster,min_dist)

def closestClusterBis(x):
    a=x[0]
    min_tuple = x[1][0]
    for i in x[1]:
	if i[1]<min_tuple[1]:
	   min_tuple=i
    return (a,min_tuple)

def sumList(x,y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenneList(x,n):
    return [x[i]/n for i in range(len(x))]

def calculateDistanceKpp(centroid, dataPoint):
    array1 = numpy.array(centroid[1][:-1])
    array2 = numpy.array(dataPoint[1][:-1])
    dist = numpy.linalg.norm(array1-array2)
    return dist

def distanceFromCentroids(data, centroids):
    DistMinArray = numpy.array([min([calculateDistanceKpp(centroid, dataPoint) for centroid in centroids]) for dataPoint in data])
    return DistMinArray

def chooseNextCentroid(d, dataArray):
    d = numpy.array(d)
    probabilities = d/d.sum()
    cumprobs = probabilities.cumsum()
    randomValue = random.random()
    centroidIndex = numpy.where(cumprobs >= randomValue)[0][0]
    centroid = dataArray[centroidIndex]
    return(centroid)

def sl(a,b):
    return [x + y for x, y in zip(a, b)]

def simpleKmeans(data, nb_clusters):
    clusteringDone = False
    number_of_steps = 0
    current_error = float("inf")
    epsilon = 0.000001

    nb_elem = sc.broadcast(data.count())

    #############################
    # Select initial centroides #
    #############################
    
    centroids = data.takeSample(False, 1)
    dataArray = data.collect()    

    while len(centroids) < nb_clusters:
        DistMinArray = distanceFromCentroids(dataArray, centroids)
        centroids.append(chooseNextCentroid(DistMinArray, dataArray))

    centroids = sc.parallelize(centroids)
    centroids = centroids.map(lambda (index, data): data[:-1])
    centroids = centroids.zipWithIndex()
    centroides = centroids.map(lambda (data, index): (index, data))

    c = sc.broadcast([list(row) for row in centroides.collect()])
    start_time = time.time()

    while not clusteringDone:

        joined = data.map(lambda x:(x,[c.value[i] for i in range(nb_clusters)]))

        min_dist = joined.map(lambda x: closestClusterBis(((x[0][0],x[0][1]),[(x[1][i][0], computeDistance(x[0][1][:-1], x[1][i][1])) for i in range(nb_clusters)])))


        min_distBis = min_dist.map(lambda x:(x[0][0],x[1]))
        min_distBis.persist()
 
        clusters = min_dist.map(lambda x: (x[1][0], (x[0][1][:-1],1)))

        count=clusters.reduceByKey(lambda x,y: (sl(x[0],y[0]), x[1]+y[1]))
        centroidesCluster=count.map(lambda x: (x[0],moyenneList(x[1][0],x[1][1])))
        	
        if number_of_steps == 0:
            switch = 150
            prev_assignment = 0
        else:
            switch = sqrt(min_distBis.map(lambda x: x[1][1]).reduce(lambda x,y: x + y))/nb_elem.value

        if abs(switch-prev_assignment)>epsilon and number_of_steps != 50:
            c = sc.broadcast([list(row) for row in centroidesCluster.collect()])
            prev_assignment = sqrt(min_distBis.map(lambda x: x[1][1]).reduce(lambda x,y: x + y))/nb_elem.value
            print("*******************************NofSteps:"+str(number_of_steps))
            print("switch"+str(switch))
            number_of_steps += 1
            
        else:
            clusteringDone = True
            error = sqrt(min_distBis.map(lambda x: x[1][1]).reduce(lambda x,y: x + y))/nb_elem.value

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Number of setps: "+str(number_of_steps))
    print("epsilon: "+str(epsilon))
    return (min_dist, error, number_of_steps,switch)


if __name__ == "__main__":
    try:
        os.system('hdfs dfs -rm -r /user/user16/data/output')
    except Exception as e:
        pass
    conf = SparkConf().setAppName('exercice')
    sc = SparkContext(conf=conf)
    sqlContext = sql.SQLContext(sc)
    lines = sc.textFile("hdfs:/user/user16/data/xgb.txt")
    data = lines.map(lambda x: x.split(','))\
            .map(lambda x: [float(i) for i in x[:4]]+[x[4]])\
            .zipWithIndex()\
            .map(lambda x: (x[1],x[0]))

    #data = data.partitionBy(12)

    clustering = simpleKmeans(data,3)
    
    clustering[0].saveAsTextFile("hdfs:/user/user16/data/output")
    print(clustering)
    
