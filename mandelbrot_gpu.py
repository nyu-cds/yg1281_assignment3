#
# Simple Python program to calculate elements in the Mandelbrot set.
#
import numpy as np
from pylab import imshow, show
from numba import cuda

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
        Given the real and imaginary parts of a complex number,
        determine if it is a candidate for membership in the
        Mandelbrot set given a fixed number of iterations.
        '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters


@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
        Calculate the mandel value for each element in the
        image array. The real and imag variables contain a
        value for each element of the complex space defined
        by the X and Y boundaries (min_x, max_x) and
        (min_y, max_y).
        '''
    height = image.shape[0]
    width = image.shape[1]
    
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    row, col = cuda.grid(2) 
    
    #block width and height
    bw, bh = cuda.blockDim.x, cuda.blockDim.y
    #grid width and height
    gw, gh = cuda.gridDim.x, cuda.gridDim.y

    step_h = int(image.shape[0] / (gw * bw))
    start_h = step_h * row
    end_h = start_h + step_h

    step_w = int(image.shape[1] / (gh*bh))
    start_w = step_w * col
    end_w = start_w + step_w

    for x in range(start_w, end_w):

        real = min_x + x * pixel_size_x
        for y in range(start_h, end_h):
            imag = min_y + row * pixel_size_y
            image[y,x] = mandel(real, imag, iters)

if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockDim = (32, 8)
    gridDim = (32, 16)
    image_global_mem = cuda.to_device(image)
    compute_mandel[gridDim, blockDim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20)
    image_global_mem.copy_to_host()

    imshow(image)
    show()
