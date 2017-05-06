'''
Calculates the product of all the numbers from 1 to 1000 and prints the result
'''

import findspark
findspark.init()
from pyspark import SparkContext
from operator import mul

sc = SparkContext()
pro = sc.parallelize(range(1, 1001)).fold(1, mul)
print("The product of all the numbers from 1 to 1000 is {}".format(pro))
