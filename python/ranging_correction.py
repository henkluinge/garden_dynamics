import numpy as np

measurements = [
                {'distance': [5, 4, 4.4, 0.4]},
                {'distance': [5, 6, 7.6, 0.4]},
                {'distance': [0, 1, 6.1-1, 0.4]},
                {'distance': [0, 2, 8.6-1, 0.4]},
                {'distance': [0, 6, 18.3-1, 0.4]},
                {'distance': [0, 7, 15.45-1, 0.4]},
                {'distance': [6, 7, 3.1, 0.4]},
                {'distance': [0, 8, 16.9-1, 0.4]},
                {'distance': [0, 9, 19.2-1, 0.4]},
                {'distance': [5, 9, 9.1, 0.4]},
                {'distance': [9, 18, 6.9, 0.4]},
                {'distance': [0, 18, 15.4-1, 0.4]},
                {'distance': [0, 10, 20.9-1, 0.4]},
                {'distance': [5, 10, 13.8, 0.4]},
                {'distance': [0, 11, 23-1, 0.4]},
                {'distance': [5, 11, 16., 0.4]},
                {'distance': [0, 12, 25.4-1, 0.4]},
                {'distance': [5, 12, 20.16, 0.4]},
                {'distance': [0, 13, 22.4-1, 0.4]},
                {'distance': [12, 13, 6.2, 0.4]},
                {'distance': [0, 17, 7.6-1, 0.4]},
                {'distance': [5, 16, 25.9, 0.4]},
                {'distance': [0, 16, 30-1, 0.4]},
                {'distance': [12, 14, 9.1, 0.4]},
                {'distance': [0, 14, 27.9-1, 0.4]},
                ]

def initialize_state_from_gps(gdf_trees, sigma_gps=4.0, sigma_corner=0.1):    
    """ 
    Initialize a state from gps measurements. Each position of each tree is in the state. Becasue of 
    2D position, the state has an x, y value for each tree.

    sigma_gps: standard deviation of GPS measurement noise (meters).
    sigma_corner: 
    """

    state_indices = {}
    for i, row in gdf_trees.iterrows():
        indices = 2*i + np.array([0,1])
        label = row['label']
        state_indices[label] = indices
    n_states = np.max(np.asarray(list(state_indices.values())).flatten()) + 1

    x = np.zeros(n_states)
    Px = np.eye(n_states)
     
    for label, state_index in state_indices.items():
        s_trees = gdf_trees[gdf_trees.label==label].iloc[0]  # Series
        # p = s_trees.geometry
        x[state_index[0]] = s_trees.px
        x[state_index[1]] = s_trees.py
        
        if s_trees.type=='Corner':
            sigma_init = sigma_corner
        else:
            sigma_init = sigma_gps
        Px[state_index[0], state_index[0]] = sigma_init**2
        Px[state_index[1], state_index[1]] = sigma_init**2
    
    return x, Px, state_indices