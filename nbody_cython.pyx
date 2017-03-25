'''
Author: Yuting Gui

Date: 2/11
Opt version running time 41.44s
Original version running time: 130.2s
R = 3.14


Date:2/18
Opt version running time: 37.19s
Original version running time:130.2s
R = 3.50

Date:3/14
Cython version runnint time: 14.07s
Original version running time: 130.2s
R = 9.25
'''
import time
"""
    N-body simulation.
"""

cdef advance(dt, body_pair, bodies):
    '''
        advance the system one timestep
    '''
    for body1, body2 in body_pair:
        ([x1, y1, z1], v1, m1) = bodies[body1]
        ([x2, y2, z2], v2, m2) = bodies[body2]
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2) #compute_deltas
        
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

    
cdef report_energy(body_pair, bodies, float e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    for body1, body2 in body_pair:
        ((x1, y1, z1), v1, m1) = bodies[body1]
        ((x2, y2, z2), v2, m2) = bodies[body2]
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2) #compute_deltas
        e -=  (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5) #compute_energy
    for body in bodies:
        (r, [vx, vy, vz], m) = bodies[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.

    return e

cdef offset_momentum(ref, bodies, float px=0.0, float py=0.0, float pz=0.0):
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


cdef nbody(int loops, reference, int iterations, bp, bodies):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    offset_momentum(bodies[reference], bodies)

    for _ in range(loops):
        report_energy(bp, bodies)
        for _ in range(iterations):
            advance(0.01, bp, bodies)
        print(report_energy(bp, bodies))

def main():
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * (3.14159265358979323**2)
    DAYS_PER_YEAR = 365.24
    
    BODIES = {
        'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),
    
        'jupiter': ([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01],
                    [1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR],
                    9.54791938424326609e-04 * SOLAR_MASS),
    
        'saturn': ([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01],
                   [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR],
                   2.85885980666130812e-04 * SOLAR_MASS),
    
        'uranus': ([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01],
                   [2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR],
                   4.36624404335156298e-05 * SOLAR_MASS),
    
        'neptune': ([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01],
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
    nbody(100, 'sun', 20000, body_pair, BODIES)
    t2 = time.time()
    print("Total time is ", (t2-t1))
