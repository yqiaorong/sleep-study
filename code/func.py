def epoching(args, data_part, seed):
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

		### Get events, drop unused channels and reject target trials ###
		events = mne.find_events(raw, stim_channel='stim')
		# Reject the target trials (event 99999)
		idx_target = np.where(events[:,2] == 99999)[0]
		events = np.delete(events, idx_target, 0)

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
		if args.sfreq < 1000:
			epochs.resample(args.sfreq)

		### Get epoched channels and times ###
		ch_names = epochs.info['ch_names']
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

def mvnn(args, epoched_test, epoched_train):
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

def save_prepr(args, whitened_test, whitened_train, img_conditions_train,
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

	import numpy as np
	from sklearn.utils import shuffle
	import os

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
	save_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data', 
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
	np.save(os.path.join(save_dir, file_name_train),
		train_dict)
	del train_dict
 
def train_model_THINGS2():
    """The function trains the encoding model using LinearRegression. X train 
    is THINGS2 dnn feature maps and Y train is the real THINGS EEG2 training 
    data.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    ----------
    reg: The trained LogisticRegression model.
    """

    import os
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    ### Load the training DNN feature maps ###
    # Load the training DNN feature maps directory
    dnn_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'dnn_feature_maps')
    # Load the training DNN feature maps (16540, 300)
    dnn_fmaps_train = np.load(os.path.join(dnn_train_dir, 'new_feature_maps.npy'), 
                            allow_pickle=True)

    ### Load the training EEG data ###
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data')
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,7), desc='THINGS EEG2 subjects'):
        # Load the THINGS2 training EEG data
        data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                    'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        # Average the training EEG data across repetitions: (16540,64,100)
        data = np.mean(data['preprocessed_eeg_data'], 1)
        # Reshape the data: (16540, 64 x 100)
        data = np.reshape(data, (data.shape[0],-1))
        # Append individual data
        eeg_data_train.append(data)
        del data
    # Average the training EEG data across subjects: (16540, 64 x 100)
    eeg_data_train = np.mean(eeg_data_train, 0)
    print('eeg_data_train shape', eeg_data_train.shape)

    ### Train the encoding model ###
    # Train the encoding models
    reg = LinearRegression().fit(dnn_fmaps_train, eeg_data_train)
    return reg