# usage: $ mpiexec -n 4 python parallel_sorter.py
from mpi4py import MPI
import numpy as np

# initialize communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def multithread_sort():
    if rank == 0:
        # when rank is 0, generate 10000 random integers, and slice it into bins
        numbers = np.random.randint(0, 100, 10000)
        min_num = min(numbers)
        max_num = max(numbers)
        num_range = (max_num - min_num)/float(size)
        all_process = []
        for i in range(size):
            all_process.append(filter(lambda x: x >= min_num + i*num_range\
                    and x <= min_num + (i+1)*num_range, numbers))
    else:
        all_process = None

    # send slice into each process
    num_scatter = comm.scatter(all_process, root=0)

    # product sort for each process
    num_scatter.sort()

    # gather the sorted array
    sorted_num = comm.gather(num_scatter, root=0)

    if rank == 0:
        # concatenate to sorted array, and assert it is sorted
        sorted_num = np.concatenate(sorted_num) 
        assert(all(sorted_num == sorted(sorted_num)))
        print("The array is being sorted")
    return sorted_num


if __name__ == '__main__':
    sorted_num = multithread_sort()

