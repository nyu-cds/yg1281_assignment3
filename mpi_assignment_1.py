from mpi4py import MPI

# get rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank%2 == 0: # even renk, print Hello
    print("Hello from process " + str(rank))
else: #odd rank, print Goodbye
    print("Goodbye from process " + str(rank))