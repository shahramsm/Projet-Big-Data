
from numpy import array
from math import sqrt
#from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
import time

def computeDistance(x,y):
    return sqrt(sum([(a - b)**2 for a,b in zip(x,y)]))

# la fonction qui calcule la somme des distances entre chaque point et son centroide associes 
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

if __name__ == "__main__":

    sc = SparkContext(appName="exercice")  # SparkContext

    # Load and parse the data
    data = sc.textFile("hdfs:/user/user16/data/iris.data.txt")
    
    data_km = data.map(lambda x: x.split(','))

    data_km = data_km.map(lambda x: [float(i) for i in x[:4]])
    parsedData = data_km.map(lambda x: array(x))
    start_time = time.time()
    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 3, maxIterations=100, epsilon=0.000001, initializationMode="k-means||")
    print(clusters.k)
    print("--- %s seconds ---" % (time.time() - start_time))

    
    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("la somme des distances = " + str(WSSSE))

    # Save and load model
    data2 = data.map(lambda x: x.split(','))\
             .map(lambda x: [float(i) for i in x[:7]])

    data3=data2.zipWithIndex().map(lambda x: (x[1],x[0]))

    data6=data3.map(lambda x : (x[0],((clusters.predict(array(x[1]))\
           ,computeDistance(x[1],clusters.centers[clusters.predict(array(x[1]))]\
           .tolist())),x[1])))

    data6.saveAsTextFile("file:///Users/plu/Desktop/output_weather1-final")
    sc.stop()









