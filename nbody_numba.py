'''
Author: Yuting Gui

Date: 4/1
Opt version running time 68.0053
jit version running time 64.988
jit+signature version runningtime: 57.869

Adding vec_delta: 104.846, it is slow down the running process, probably because of the array dimension is too small
'''
from numba import jit, int64, float32, float64, vectorize
import numpy as np
import time
"""
    N-body simulation.
"""


@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y


def advance(dt, body_pair):
    '''
        advance the system one timestep
    '''
    for body1, body2 in body_pair:
        (xyz1, v1, m1) = bodies[body1]
        (xyz2, v2, m2) = bodies[body2]
        (dx, dy, dz) = vec_deltas(xyz1, xyz2) #compute_deltas
        
        # update vs
        b = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5)) 
        b2 = b * m2
        b1 = b * m1
        v1[0] -= dx * b2
        v1[1] -= dy * b2 
        v1[2] -= dz * b2 
        v2[0] += dx * b1 
        v2[1] += dy * b1
        v2[2] += dz * b1
        
    for body in bodies:
        (r, [vx, vy, vz], m) = bodies[body]
        #update_rs
        r[0] += dt * vx
        r[1] += dt * vy
        r[2] += dt * vz

    
def report_energy(body_pair, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    for body1, body2 in body_pair:
        (xyz1, v1, m1) = bodies[body1]
        (xyz2, v2, m2) = bodies[body2]
        (dx, dy, dz) = vec_deltas(xyz1, xyz2) #compute_deltas
        e -=  (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5) #compute_energy
    for body in bodies:
        (r, [vx, vy, vz], m) = bodies[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.

    return e

def offset_momentum(ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    for body in bodies:
        (r, [vx, vy, vz], m) = bodies[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m

# @jit
@jit('void(int64,int64)')
def nbody(loops, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
        '''
    # Set up global state
    offset_momentum(bodies['sun'])
    
    for _ in range(loops):
        report_energy(body_pair)
        for _ in range(iterations):
            advance(0.01, body_pair)
        print(report_energy(body_pair))

if __name__ == '__main__':
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * (3.14159265358979323**2)
    DAYS_PER_YEAR = 365.24
    
    bodies = {
        'sun': (np.array([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], SOLAR_MASS),
    
        'jupiter': (np.array([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01]),
                    [1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR],
                    9.54791938424326609e-04 * SOLAR_MASS),
    
        'saturn': (np.array([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01]),
                   [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR],
                   2.85885980666130812e-04 * SOLAR_MASS),
    
        'uranus': (np.array([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01]),
                   [2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR],
                   4.36624404335156298e-05 * SOLAR_MASS),
    
        'neptune': (np.array([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01]),
                    [2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR],
                    5.15138902046611451e-05 * SOLAR_MASS)}


    body_pair = [('sun','saturn'),
            ('sun','jupiter'),
            ('sun','neptune'),
            ('sun','uranus'),
            ('saturn','jupiter'),
            ('saturn','neptune'),
            ('saturn','uranus'),
            ('jupiter','neptune'),
            ('jupiter','uranus'),
            ('neptune','uranus')]

    t1 = time.time()
    nbody(100,  20000)
    t2 = time.time()
    print("Total time is ", (t2-t1))
