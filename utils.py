#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

def compute_ev_ew(H, k):
    """ Computing first k EV and EW of symmetric H """
    print('Computing SVD')
    if isinstance(H, np.ndarray):
        ev, ew, _ = np.linalg.svd(H)
        ev = ev[:,:k]
        ew = ew[:k]
    else:
        ev, ew, _ = svds(H, k=k,return_singular_vectors='u')
        ev = np.flip(ev, axis=1)
        ew = np.flip(ew)
    return ev, ew


def load_array(array, nb, filename, name = ''):
    """ From single file   """
    try:
        computed = np.load(filename)
        nb_loaded = min(np.shape(computed)[0], nb)
        array[:nb_loaded] = computed[:nb_loaded]
        print(f'Loaded {nb_loaded} {name} from {filename}')
    except FileNotFoundError:
        nb_loaded = 0
        print(f'No existing {name} in {filename}')
    return nb_loaded, array

def load_multiple_array(array, nb, filename, name = ''):
    """ From multiple files    """
    nb_loaded = 0
    for i in range(nb):
        try:
            array[i] = np.load(f'{filename}_{i}.npy')
            nb_loaded = i+1 
        except:
            break
    if nb_loaded == 0:
        print(f'No existing {name} in  {filename}_i.npy')
    else:
        print(f'Loaded {nb_loaded} {name} from {filename}_i.npy')
    return nb_loaded, array
    