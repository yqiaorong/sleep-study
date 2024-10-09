def load_full_fmaps(args, p='training'):
	import os
	import numpy as np
	from tqdm import tqdm
 
	feats = []
	final_fmaps = {}
	# The directory of the dnn training feature maps
	fmaps_dir = os.path.join('dataset','THINGS_EEG2','dnn_feature_maps',
							'full_feature_maps', 'Alexnet', 
							'pretrained-'+str(args.pretrained),
							f'{p}_images')
	fmaps_list = os.listdir(fmaps_dir)
	fmaps_list.sort()
	for f, fmaps in enumerate(tqdm(fmaps_list, desc='load THINGS training images')):
		fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
								allow_pickle=True).item()
		all_layers = fmaps_data.keys()
		for l, dnn_layer in enumerate(all_layers):
			if f == 0:
				feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
			else:
				feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
		
	final_fmaps[args.layer_name] = np.squeeze(np.asarray(feats[l]))
	print(f'The original {p} fmaps shape', final_fmaps[args.layer_name].shape)

	return final_fmaps

def train_model_THINGS2(args):
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
    # Load the training DNN feature maps (img, num_feat)
    dnn_fmaps_train = np.load(os.path.join(dnn_train_dir, 
                                           f'new_feature_maps_{args.num_feat}{args.adapt_to}.npy'), 
                              allow_pickle=True)

    ### Load the training EEG data ###
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data'+args.adapt_to)
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,11), desc='load THINGS EEG2 subjects'):
        # Load the THINGS2 training EEG data
        if args.adapt_to == '':
            data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                    'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        else:
            data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                    'preprocessed_eeg_training.npy'), allow_pickle=True)
        # Average the training EEG data across repetitions: (img, ch, time)
        data = np.mean(data['preprocessed_eeg_data'], 1)
        # Drop the stim channel: (img, ch, time)
        data = np.delete(data, -1, axis=1)
        # Reshape the data: (img, ch x time)
        data = np.reshape(data, (data.shape[0],-1))
        # Append individual data
        eeg_data_train.append(data)
        del data
    # Average the training EEG data across subjects: (img, ch x time)
    eeg_data_train = np.mean(eeg_data_train, 0)
    print('eeg_data_train shape', eeg_data_train.shape)

    ### Train the encoding model ###
    # Train the encoding models
    reg = LinearRegression().fit(dnn_fmaps_train, eeg_data_train)
    return reg

def corr(args, pred_eeg_data_test, test_subj):
    import os
    import numpy as np
    from scipy.stats import pearsonr as corr

    ### Load the test EEG data ###
    # Load the THINGS1 test EEG data 
    eeg_test_dir = os.path.join('dataset', args.test_dataset, 'preprocessed_data', 
                                'sub-'+format(test_subj,'02'))
    eeg_data_test = np.load(os.path.join(eeg_test_dir, 'preprocessed_eeg_test.npy'),
                            allow_pickle=True).item()
    # Get the number of test images
    num_img = eeg_data_test['preprocessed_eeg_data'].shape[0]
    # Get test channel names and times
    num_ch = 63
    test_times = eeg_data_test['times']
    # Average the test EEG data across repetitions if it's THINGS
    eeg_data_test_avg = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)
    # Drop the stim channel of THINGS EEG2:(200,63,100)
    if args.test_dataset == 'THINGS_EEG2':
        eeg_data_test_avg = np.delete(eeg_data_test_avg,-1,axis=1)
        
    ### Separate the dimension of EEG channels and times ###
    pred_eeg_data_test = np.reshape(pred_eeg_data_test,(num_img,num_ch,len(test_times)))
    eeg_data_test_avg = np.reshape(eeg_data_test_avg,(num_img,num_ch,len(test_times)))
    del eeg_data_test
    
    ### Test the encoding model ###
    # Calculate the encoding accuracy
    encoding_accuracy = np.zeros((num_ch,len(test_times)))
    for t in range(len(test_times)):
        for c in range(num_ch):
            encoding_accuracy[c,t] = corr(pred_eeg_data_test[:,c,t],
                eeg_data_test_avg[:,c,t])[0]
    # Average the encoding accuracy across channels
    encoding_accuracy = np.mean(encoding_accuracy,0)
            
    # ### Output ###
    return encoding_accuracy, test_times