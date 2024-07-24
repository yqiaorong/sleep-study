def mvnn(all_epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch of each image condition), and then average them 
    across image conditions. The inverse of the resulting averaged covariance
    matrix is used to whiten the EEG data.

    Parameters
    ----------
    epoched_data : list of arrays of shape (rep,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : list of arrays of shape (rep,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Compute the covariance matrices ###
    tot_sigma = []
    for data in tqdm(all_epoched_data): # Iterate over imgs
        
        # Notations
        num_rep = data.shape[0]
        num_ch = data.shape[1]
        num_time = data.shape[2]
        
        # Covariance matrix of shape (EEG channels × EEG channels)
        sigma = np.mean([_cov(data[:,:,t], shrinkage='auto') for t in range(num_time)],
                        axis=0)
        tot_sigma.append(sigma)
        
    # Average the covariance matrices across image conditions
    tot_sigma = np.array(tot_sigma)
    mean_sigma = np.mean(tot_sigma, axis=0)

    # # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(mean_sigma, -0.5)
  
    ### Whiten the data ###
    whitened_data = []
    for data in all_epoched_data: # Iterate over imgs     
        whitened_data.append((data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2))

    ### Output ###  
    return whitened_data