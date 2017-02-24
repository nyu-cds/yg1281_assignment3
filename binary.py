from itertools import *



def bits(n=None, k=None):
    '''
    Input: n: the length of binary string
           k: k zero bits in the binary string
    Return: a set of strings
    '''
    # raise error if k and n are not being initialized
    if n == None and k == None:
        raise ValueError("Initialize n and k")

    # raise error if k or n is not an integer
    if int(n) != n or int(k) != k:
        raise ValueError("k and n should be integers")
    

    
    res = set() 
    # use combinations to get the positions of "0"
    for tuple in combinations(range(n), k):
        temp = ["1"] * n
        for i in tuple:
            temp[i] = "0"
        res.add("".join(temp))
    return res 


'''
assert bits(4, 3) == {'0100', '0001', '0010', '1000'}
assert bits(4, 1) == {'0111', '1011', '1101', '1110'}
assert bits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}
'''
