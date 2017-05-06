'''
calculate the average of the square root of all the numbers from 1 to 1000.
i.e the sum of the square roots of all the numbers divided by 1000.
'''

import findspark
findspark.init()
from pyspark import SparkContext
from operator import add, pow

sqrt = sc.parallelize(range(1, 1001)).map(lambda x: pow(x, 0.5))
avg_sqrt = sqrt.fold(0, add) / float(sqrt.count())
