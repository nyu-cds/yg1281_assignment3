import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

myint = np.zeros(1, dtype=int)

if rank == 0:
    # if rank is 0, ask user to input an integer
    print("Input an integer less than 100:")
    num = raw_input()
    
    # if the integer is greater than 100, ask for another
    while int(num) >= 100:
        print("Try other integer less than 100:")
        num = raw_input()
    
    # send the integer to process 1, and receive the output from the last process
    myint[0] = num
    comm.Send(myint, dest=1)
    comm.Recv(myint, source=size-1)
    print("The final output is " + str(myint[0]))

elif rank < size-1:
    # if this is not the first and not the last process, multiply the integer by its rank, and send to next process
    comm.Recv(myint, source=rank-1)
    myint = myint*rank
    comm.Send(myint, dest=rank+1)
else:
    # if this is the last process, multiply the integer by its rank, and send to the first process
    comm.Recv(myint, source=rank-1)
    myint = myint*rank
    comm.Send(myint, dest=0)

