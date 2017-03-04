'''
Add function:
    The two for loops are the most time consuming parts.
    Change the for loops to add 
Multiply function:
    The two for loops are the most time consuming parts.
    Change them to x*y
Sqrt function:
    The two for loops are the most time consuming parts.
    Change them to np.sqrt
Hypotheuse:
    Use new function to implement hypotheuse
'''
import numpy as np
def add(x,y):
    """
    Add two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return x+y


def multiply(x,y):
    """
    Multiply two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return x*y


def sqrt(x):
    """
    Take the square root of the elements of an arrays using a Python loop.
    """
    return np.sqrt(x)


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    return sqrt(add(multiply(x, x), multiply(y, y)))
