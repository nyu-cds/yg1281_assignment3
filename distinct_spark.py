'''
Calculate distinct keys in the text
python2: please uncomment line 17
python3: leave it commented
'''


# if unable to find pyspark, please install findspark and uncomment the following two lines
# import findspark
# findspark.init()

import re
from pyspark import SparkContext

def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    # line = line.encode('ascii','ignore')
    return map(str.lower, re.split(r'\W+', line))


if __name__ == '__main__':
    sc = SparkContext("local", "wordcount")
    text = sc.textFile("pg2701.txt")
    words = text.flatMap(splitter)
    words_mapped = words.map(lambda x: (x, 1))

    # count distinct keys
    distinct_count = words_mapped.keys().distinct().count()
    print("The number of distinct words in the input text is {}.".format(distinct_count))

