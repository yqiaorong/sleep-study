def epoching_THINGS2(args, data_part, seed):
    """This function first converts the EEG data to MNE raw format, and
    performs re-reference, bandpass filter, epoching, baseline correction 
    and frequency downsampling. Then, it sorts the EEG data of each session
        according to the image conditions.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    data_part : str
        'test' or 'training' data partitions.
    seed : int
        Random seed.

    Returns
    -------
    epoched_data : list of float
        Epoched EEG data.
    img_conditions : list of int
        Unique image conditions of the epoched and sorted EEG data.
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.

    """

    import os
    import mne
    import numpy as np
    from sklearn.utils import shuffle

    ### Loop across data collection sessions ###
    epoched_data = []
    img_conditions = []
    for s in range(args.n_ses):

        ### Load the EEG data and convert it to MNE raw format ###
        eeg_dir = os.path.join('dataset', 'THINGS_EEG2', 'raw_data', 
                            'sub-'+format(args.subj,'02'), 
                            'ses-'+format(s+1,'02'), 'raw_eeg_'+data_part+'.npy')
        eeg_data = np.load(eeg_dir, allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info)
        del eeg_data
        
        ### channel selection ###
        if args.adapt_to == '_sleemory':
        # Pick the main 56 sleemory channels (sleemory rejects 'Fpz' and 'Fz' and 
        # keeps 'Cz')
            channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 
                        'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Cz', 'CPz', 
                        'Pz', 'Oz', 'AF3', 'AF4', 'PO3', 'PO4', 'AF7', 'AF8', 
                        'F1', 'F2', 'F5', 'F6', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 
                        'FC6', 'FT7', 'FT8', 'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 
                        'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'P1', 'P2', 
                        'P5', 'P6', 'PO7', 'PO8', 'stim']
            raw = raw.pick(channels)

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim')
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        ### Re-reference and bandpass filter all channels ###
        # Re-reference raw 'average'
        raw.set_eeg_reference()
        # Bandpass filter
        if args.adapt_to == '_sleemory':
            raw.filter(l_freq=0.01, h_freq=30, n_jobs=2)
        else:
            raw.filter(l_freq=0.1, h_freq=100, n_jobs=-1)

        ### Epoching, baseline correction and resampling ###
        # Epoching
        epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
            preload=True)
        del raw
        # Resampling
        if args.sfreq < 1000:
            epochs.resample(args.sfreq)

        ### Get epoched channels and times ###
        ch_names = epochs.info['ch_names']
        print(len(ch_names))
        times = epochs.times

        ### Sort the data ###
        data = epochs.get_data(copy=False)
        events = epochs.events[:,2]
        img_cond = np.unique(events)
        del epochs
        # Select only a maximum number of EEG repetitions
        if data_part == 'test':
            max_rep = 20
        else:
            max_rep = 2
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
            data.shape[2]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)
        del sorted_data

    ### Output ###
    return epoched_data, img_conditions, ch_names, times

def mvnn_THINGS2(args, epoched_test, epoched_train):
	"""Compute the covariance matrices of the EEG data (calculated for each
	time-point or epoch/repetitions of each image condition), and then average
	them across image conditions and data partitions. The inverse of the
	resulting averaged covariance matrix is used to whiten the EEG data
	(independently for each session).

	Parameters
	----------
	args : Namespace
		Input arguments.
	epoched_test : list of floats
		Epoched test EEG data.
	epoched_train : list of floats
		Epoched training EEG data.

	Returns
	-------
	whitened_test : list of float
		Whitened test EEG data.
	whitened_train : list of float
		Whitened training EEG data.

	"""

	import numpy as np
	from tqdm import tqdm
	from sklearn.discriminant_analysis import _cov
	import scipy

	### Loop across data collection sessions ###
	whitened_test = []
	whitened_train = []
	for s in range(args.n_ses):
		session_data = [epoched_test[s], epoched_train[s]]

		### Compute the covariance matrices ###
		# Data partitions covariance matrix of shape:
		# Data partitions × EEG channels × EEG channels
		sigma_part = np.empty((len(session_data),session_data[0].shape[2],
			session_data[0].shape[2]))
		for p in range(sigma_part.shape[0]):
			# Image conditions covariance matrix of shape:
			# Image conditions × EEG channels × EEG channels
			sigma_cond = np.empty((session_data[p].shape[0],
				session_data[0].shape[2],session_data[0].shape[2]))
			for i in tqdm(range(session_data[p].shape[0])):
				cond_data = session_data[p][i]
				# Compute covariace matrices at each time point, and then
				# average across time points
				if args.mvnn_dim == "time":
					sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
						shrinkage='auto') for t in range(cond_data.shape[2])],
						axis=0)
				# Compute covariace matrices at each epoch (EEG repetition),
				# and then average across epochs/repetitions
				elif args.mvnn_dim == "epochs":
					sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
						shrinkage='auto') for e in range(cond_data.shape[0])],
						axis=0)
			# Average the covariance matrices across image conditions
			sigma_part[p] = sigma_cond.mean(axis=0)
		# Average the covariance matrices across image partitions
		sigma_tot = sigma_part.mean(axis=0)
		# Compute the inverse of the covariance matrix
		sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

		### Whiten the data ###
		whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
			session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
			@ sigma_inv).swapaxes(1, 2), session_data[0].shape))
		whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
			session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
				@ sigma_inv).swapaxes(1, 2), session_data[1].shape))

	### Output ###
	return whitened_test, whitened_train

def save_prepr_THINGS2(args, whitened_test, whitened_train, img_conditions_train,
	ch_names, times, seed):
    """Merge the EEG data of all sessions together, shuffle the EEG repetitions
    across sessions and reshaping the data to the format:
    Image conditions × EGG repetitions × EEG channels × EEG time points.
    Then, the data of both test and training EEG partitions is saved.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    whitened_test : list of float
        Whitened test EEG data.
    whitened_train : list of float
        Whitened training EEG data.
    img_conditions_train : list of int
        Unique image conditions of the epoched and sorted train EEG data.
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.
    seed : int
        Random seed.

    """
    import os
    import pickle
    import numpy as np
    from sklearn.utils import shuffle

    ### Merge and save the test data ###
    for s in range(args.n_ses):
        if s == 0:
            merged_test = whitened_test[s]
        else:
            merged_test = np.append(merged_test, whitened_test[s], 1)
    del whitened_test
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:,idx]
    # Insert the data into a dictionary
    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names,
        'times': times
    }
    del merged_test
    # Saving directories
    save_dir = os.path.join('dataset', 'THINGS_EEG2', f'preprocessed_data{args.adapt_to}', 
                            'sub-'+format(args.subj,'02'))
    file_name_test = 'preprocessed_eeg_test.npy'
    file_name_train = 'preprocessed_eeg_training.npy'
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_test), test_dict)
    del test_dict

    ### Merge and save the training data ###
    for s in range(args.n_ses):
        if s == 0:
            white_data = whitened_train[s]
            img_cond = img_conditions_train[s]
        else:
            white_data = np.append(white_data, whitened_train[s], 0)
            img_cond = np.append(img_cond, img_conditions_train[s], 0)
    del whitened_train, img_conditions_train
    # Data matrix of shape:
    # Image conditions × EGG repetitions × EEG channels × EEG time points
    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
        white_data.shape[2],white_data.shape[3]))
    for i in range(len(np.unique(img_cond))):
        # Find the indices of the selected category
        idx = np.where(img_cond == i+1)[0]
        for r in range(len(idx)):
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
        merged_train[i] = ordered_data
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:,idx]
    # Insert the data into a dictionary
    train_dict = {
        'preprocessed_eeg_data': merged_train,
        'ch_names': ch_names,
        'times': times
    }
    del merged_train
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    if args.adapt_to == '_sleemory':
        with open(os.path.join(save_dir, file_name_train), 'wb') as f: 
            pickle.dump(train_dict, f, protocol=4) 
    else:
        np.save(os.path.join(save_dir, file_name_train), train_dict)
    del train_dict

def epoching_THINGS1(args):
    """The function preprocesses the raw EEG file: channel selection, 
    creating annotations and events, re-reference, bandpass filter,
    epoching, baseline correction and frequency downsampling. Then, it 
    sorts the test EEG data according to the image conditions.

    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    -------
    sort_data : array of shape (image,repetition,channel,time)
        Epoched EEG test data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.
    """

    import os
    import mne
    import numpy as np
    import pandas as pd
    
    ### Load the THINGS1 subject metadata ### 
    # Load the THINGS1 subject directory
    TH1_dir = os.path.join('dataset','THINGS_EEG1','raw_data',
                           'sub-'+format(args.subj,'02'),'eeg')
    # Load the THINGS1 subject metadata
    dftsv = pd.read_csv(os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                                     '_task-rsvp_events.tsv'), delimiter='\t')
    
    ### Crop the THINGS1 subject metadata ###
    # Select the main 22248 images
    dftsv = dftsv.iloc[:22248]
    # Select events relevant information
    dftsv = dftsv[['onset','object']] 
    
    ### Load the THINGS1 subject EEG data ###
    # Load the THINGS1 subject EEG directory
    TH1_EEG_dir = os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                               '_task-rsvp_eeg.vhdr')
    # Load the THINGS1 subject EEG raw file
    raw = mne.io.read_raw_brainvision(TH1_EEG_dir, preload=True)
    
    ### channel selection ###
    # Pick the main 63 channels (THINGS EEG1 has 'Fz' while THINGS EEG2 has 'Cz')
    channels = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 
    'CP2', 'Fz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
    'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 
    'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 
    'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 
    'AF4', 'AF8']
    raw = raw.pick(channels)
    
    ### Create annotations and events ###
    # Annotation onset
    onset = dftsv['onset'] # in seconds
    # Annotation duration
    duration = [0.05]*len(dftsv) # in seconds, too
    # Create annotations
    annot = mne.Annotations(onset=onset, duration=duration, 
                            description=['images']*len(dftsv))
    # Set annotations
    raw.set_annotations(annot)
    # Create events
    events, _ = mne.events_from_annotations(raw)
    
    ### Re-reference and bandpass filter all channels ###
    # Re-reference raw 'average'
    raw.set_eeg_reference()  
    # Bandpass filter
    raw.filter(l_freq=0.1, h_freq=100)
    
    ### Epoching, baseline correction and resampling ###
    # Epoching
    epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0), 
                        preload=True)
    del raw
    # Resampling
    epochs.resample(args.sfreq)
    
    ### Get epoched channels and times ###
    ch_names = epochs.info['ch_names']
    times = epochs.times
    
    ### Sort epoched data according to the THINGS2 test images ###
    # Get epoched data
    epoched_data = epochs.get_data()
    del epochs
    # THINGS2 test images directory
    test_img_dir = os.path.join('dataset','THINGS_EEG2','image_set',
                                'test_images')
    # Create list of THINGS2 test images
    test_imgs = os.listdir(test_img_dir)
    # The sorted epoched data
    sort_data = []
    # Iterate over THINGS2 test images
    for test_img in test_imgs:
        # Get the indices of test image 
        indices = dftsv.index[dftsv['object'] == test_img[6:]]
        # Get the data of test image 
        data = [epoched_data[i, :, :] for i in indices]
        # Convert list to array
        data = np.array(data)
        # Add the data to the test THINGS1 EEG data
        sort_data.append(data)
        del indices, data
    # Convert list to array
    sort_data = np.array(sort_data)

    ### Outputs ###
    return sort_data, ch_names, times

def mvnn_THINGS1(args, epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch of each image condition), and then average them 
    across image conditions. The inverse of the resulting averaged covariance
    matrix is used to whiten the EEG data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    epoched_data : array of shape (image,repetition,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    whitened_data = []
    # Notations
    img_cond = epoched_data.shape[0]
    num_rep = epoched_data.shape[1]
    num_ch = epoched_data.shape[2]
    num_time = epoched_data.shape[3]

    ### Compute the covariance matrices ###
    # Covariance matrix of shape:
    # EEG channels × EEG channels
    sigma = np.empty((num_ch, num_ch))
    # Image conditions covariance matrix of shape:
    # Image conditions × EEG channels × EEG channels
    sigma_cond = np.empty((img_cond, num_ch, num_ch))
    # Iterate across the time points
    for i in tqdm(range(img_cond)):
        cond_data = epoched_data[i]
        # Compute covariace matrices at each time point, and then
        # average across time points
        if args.mvnn_dim == "time":
            sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                shrinkage='auto') for t in range(num_time)],
                axis=0)
        # Compute covariace matrices at each epoch, and then 
        # average across epochs
        elif args.mvnn_dim == "epochs":
            sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                shrinkage='auto') for e in range(num_rep)],
                axis=0)
    # Average the covariance matrices across image conditions
    sigma = sigma_cond.mean(axis=0)
    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

    ### Whiten the data ###
    whitened_data = np.reshape((np.reshape(epoched_data, 
    (-1,num_ch,num_time)).swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2), 
    epoched_data.shape)

    ### Output ###  
    return whitened_data

def save_prepr_THINGS1(args, whitened_data, ch_names, times, seed):
    """Shuffle the EEG repetitions across sessions and reshaping the 
    data to the format:
    Image conditions × EGG repetitions × EEG channels × EEG time points.
    Then, the data of both test EEG partitions is saved.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.
    seed : int
        Random seed.

    """
    
    import os
    import numpy as np
    from sklearn.utils import shuffle
    
    ### Save the data ###
    # Notation
    num_rep = whitened_data.shape[1]
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, num_rep), random_state=seed)
    whitened_data = whitened_data[:,idx]
    # Insert the data into a dictionary
    data_dict = {
        'preprocessed_eeg_data': whitened_data,
        'ch_names': ch_names,
        'times': times
    }
    del whitened_data
    # Saving directories
    save_dir = os.path.join('dataset','THINGS_EEG1','preprocessed_data',
                            'sub-'+format(args.subj,'02'))
    file_name_test = 'preprocessed_eeg_test.npy'
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_test), data_dict)
    del data_dict
