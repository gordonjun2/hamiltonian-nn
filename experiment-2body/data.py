# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
solve_ivp = scipy.integrate.solve_ivp
from scipy.io import loadmat
import torch
import math

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle

M_earth = 5.9742e+24
G = 6.67384e-11

##### ENERGY #####
def potential_energy(state):
    '''U=sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1,state.shape[0]):
            r_ij = ((state[i:i+1,1:3] - state[j:j+1,1:3])**2).sum(1, keepdims=True)**.5
            # r_ij is the distance between the two bodies
            m_i = state[i:i+1,0:1]
            # m_i is the mass of the first body
            m_j = state[j:j+1,0:1]
            # m_j is the mass of the second body
    tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=sum_i .5*m*v^2'''
    energies = .5 * state[:,0:1] * (state[:,3:5]**2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)

# let Earth's potential energy = 0 J
def potential_energy_over_M_sat(state):
    '''U=sum_i,j>i G m_i m_j / r_ij'''
    pe_over_M_sat = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1,state.shape[0]):
            r_ij = ((state[i:i+1,0:3] - state[j:j+1,0:3])**2).sum(1, keepdims=True)**.5
            # r_ij is the distance between the two bodies

    pe_over_M_sat += G * M_earth / r_ij
    U_over_M_sat = -pe_over_M_sat.sum(0).squeeze()
    return U_over_M_sat

# let Earth's kinetic energy = 0 J
def kinetic_energy_over_M_sat(state):
    '''T=sum_i .5*m*v^2'''
    ke_over_M_sat = .5 * (state[:,3:6]**2).sum(1, keepdims=True)
    T_over_M_sat = ke_over_M_sat.sum(0).squeeze()
    return T_over_M_sat

def total_energy_over_M_sat(state):
    return potential_energy_over_M_sat(state) + kinetic_energy_over_M_sat(state)

##### DYNAMICS #####
def get_accelerations(state, epsilon=0):
    # shape of state is [bodies x properties]
    net_accs = [] # [nbodies x 2]
    for i in range(state.shape[0]): # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 1:3] - state[i, 1:3] # indexes 1:3 -> pxs, pys
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        masses = other_bodies[:, 0:1] # index 0 -> mass
        pointwise_accs = masses * displacements / (distances**3 + epsilon) # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update(t, state):
    state = state.reshape(-1,5) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,1:3] = state[:,3:5] # dx, dy = vx, vy
    deriv[:,3:5] = get_accelerations(state)
    return deriv.reshape(-1)

def get_accelerations_satellite(state, epsilon=0):
    # shape of state is [bodies x properties]
    net_accs = [] # [nbodies x 2]
    for i in range(state.shape[0]): # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 0:3] - state[i, 0:3] # indexes 1:4 -> pxs, pys, pzs
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        #masses = other_bodies[:, 0:1]
        if i == 0:
            masses = [[M_earth]]
        else:
            # let satellite to have a mass of 0 kg here to prevent the Earth from moving in the simulation
            masses = [[0]]              
        pointwise_accs = masses * displacements / (distances**3 + epsilon) # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update_satellite(t, state):
    state = state.reshape(-1,6) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,0:3] = state[:,3:6] # dx, dy = vx, vy
    deriv[:,3:6] = get_accelerations_satellite(state)
    return deriv.reshape(-1)


##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0,2], **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-9

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    orbit = path['y'].reshape(nbodies, 5, t_points)
    return orbit, orbit_settings


##### INITIALIZE THE TWO BODIES #####
def random_config(orbit_noise=5e-2, min_radius=0.5, max_radius=1.5):
    state = np.zeros((2,5))
    state[:,0] = 1
    pos = np.random.rand(2) * (max_radius-min_radius) + min_radius
    r = np.sqrt( np.sum((pos**2)) )

    # velocity that yields a circular orbit
    vel = np.flipud(pos) / (2 * r**1.5)
    vel[0] *= -1
    vel *= 1 + orbit_noise*np.random.randn()

    # make the circular orbits SLIGHTLY elliptical
    state[:,1:3] = pos
    state[:,3:5] = vel
    state[1,1:] *= -1
    return state


##### HELPER FUNCTION #####
def coords2state(coords, nbodies=2, mass=1):
    timesteps = coords.shape[0]
    state = coords.T
    state = state.reshape(-1, nbodies, timesteps).transpose(1,0,2)
    mass_vec = mass * np.ones((nbodies, 1, timesteps))
    state = np.concatenate([mass_vec, state], axis=1)
    return state


##### INTEGRATE AN ORBIT OR TWO #####
def sample_orbits(timesteps=50, trials=1000, nbodies=2, orbit_noise=5e-2,
                  min_radius=0.5, max_radius=1.5, t_span=[0, 20], verbose=False, **kwargs):
    
    # timestep = no. of sample points in a trial (trajectory)
    # trials = trajectories
    
    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 2-body orbits:")
    
    x, dx, e = [], [], []
    N = timesteps*trials
    
    # 'while' loop loops 'trials' times
    while len(x) < N:

        # Initialize initial state
        state = random_config(orbit_noise, min_radius, max_radius)
        # state is in (mass, position in x, position in y, momentum in x, momentum in y) x 2 rows (bodies)
        
        # Calculate the full trajectory of the bodies within the time period of t_span
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, **kwargs)
        
        # Reshaped 'orbit' to be processed later
        batch = orbit.transpose(2,0,1).reshape(-1, orbit.shape[0] * orbit.shape[1])

        # 'for' loop loops 'timesteps' times
        for state in batch:
            # Calculate instantaneous velocity and acceleration
            dstate = update(None, state)
            
            # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
            coords = state.reshape(nbodies,5).T[1:].flatten()
            dcoords = dstate.reshape(nbodies,5).T[1:].flatten()
            
            # Save state in both non-canonical and canonical states
            x.append(coords)
            dx.append(dcoords)

            # Reshape state back to original form for energy calculation
            shaped_state = state.copy().reshape(2,5,1)
            
            # Calculates energy at current timestep
            e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N] }
    return data, orbit_settings

def sgp4_generated_orbits(data_percentage_usage, verbose=False, **kwargs):
    
    if verbose:
        print("Retrieving the dataset of LEO SGP4 orbits ... \n")
        
    data_root_dir = './Data_matlab/'
    raw = loadmat(os.path.join(data_root_dir, 'data_GT_cell_100_ts.mat'))
    
    raw_data = raw['data_GT_cell'][0]
    indexes = torch.randperm(raw_data.shape[0])[:math.floor(raw_data.shape[0]*data_percentage_usage)]
    raw_data = raw_data[indexes]

    x, dx, e = [], [], []

    earth_state = np.zeros((1, 6))

    trajectory_count = 0

    for trajectory in raw_data:
        trajectory_count = trajectory_count + 1
        print('Processing trajectory no. ' + str(trajectory_count) + '/' + str(raw_data.shape[0]))
        first_state_flag = 1
        for satellite_state in trajectory:
            if first_state_flag == 1:
                satellite_state = np.reshape(satellite_state, (1, -1))
                state = np.transpose(np.stack((satellite_state, earth_state)), (1, 0, 2))
                orbit = state
                first_state_flag = 0
            else:
                satellite_state = np.reshape(satellite_state, (1, -1))
                state = np.transpose(np.stack((satellite_state, earth_state)), (1, 0, 2))
                orbit = np.concatenate((orbit, state))

        batch = orbit.reshape(-1, orbit.shape[1] * orbit.shape[2])

        for state in batch:
            # Calculate instantaneous velocity and acceleration
            dstate = update_satellite(None, state)

            # reshape from [nbodies, state] where state=[m, qx, qy, qz, px, py, pz]
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, qz1, qz2, px1, px2,....]
            coords = state.reshape(2,6).T[:].flatten()
            dcoords = dstate.reshape(2,6).T[:].flatten()

            # Save state in both non-canonical and canonical states
            x.append(coords)
            dx.append(dcoords)

            # Reshape state back to original form for energy calculation
            shaped_state = state.copy().reshape(2,6,1)

            # Calculates energy at current timestep
            e.append(total_energy_over_M_sat(shaped_state))

    data = {'coords': np.stack(x),
            'dcoords': np.stack(dx),
            'energy': np.stack(e)}
    
    return data


##### MAKE A DATASET #####
def make_orbits_dataset(satellite_problem, data_percentage_usage, test_split=0.2, **kwargs):
    if not satellite_problem:
        data, orbit_settings = sample_orbits(**kwargs)
    else:
        data = sgp4_generated_orbits(data_percentage_usage, **kwargs)
    
    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
        # Each 'k' is a key in 'data'
        
    data = split_data
    
    if not satellite_problem:
        data['meta'] = orbit_settings

    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, satellite_problem, data_percentage_usage, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    if not satellite_problem:
        path = '{}/{}-orbits-dataset.pkl'.format(save_dir, experiment_name)
    else:
        path = '{}/{}-satellite-orbits-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_orbits_dataset(satellite_problem, data_percentage_usage, **kwargs)
        to_pickle(data, path)

    return data