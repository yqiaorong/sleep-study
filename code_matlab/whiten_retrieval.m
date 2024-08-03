% List of image names
img_dir = 'dataset/sleemory_retrieval/image_set';
img_files = dir(fullfile(img_dir, '*.jpg'));
imgs_names = cellfun(@(x) x(1:end-4), {img_files.name}, 'UniformOutput', false);

    for sub = 5:26
        disp(sub)
        if sub == 17
            continue;
        end
        
        % Load the test EEG data
        eeg_dir = sprintf('dataset/sleemory_retrieval/preprocessed_data/');
        data = load(fullfile(eeg_dir, sprintf('sleemory_retrieval_dataset_sub-%03d.mat', sub)));
        eegs_sub = data.ERP_all; % (1, 2)
        imgs_sub = data.imgs_all; % (1, 2)
        clear data;

        sorted_eeg_all = cell(1,2); % Final data of two sessions
        for ses = 1:2
            eegs_ses = eegs_sub{1, ses}; % (num_trials, num_ch, num_time)
            imgs_ses = imgs_sub{1, ses};

            % Classify EEG data according to image names
            whitened_data_re = nan(size(eegs_ses)); % (num_trials(100), num_ch, num_time)

            true_indices = cell(length(imgs_names), 1);
            eegs = cell(length(imgs_names), 1);
            for i = 1:length(imgs_names)

                name = imgs_names{i};
                mask = strcmp(imgs_ses, name);

                % Mark the index
                true_idx = find(mask);
                true_indices{i} = true_idx;

                % Extract the EEG
                eeg = eegs_ses(mask, :, :); % (num_trials_per_img, num_ch, num_time)
                eegs{i} = eeg;
            end

            % Whiten the data
            whitened_eegs = mvnn(eegs);
            
            for i = 1:length(imgs_names)
                whitened_eegs_re(true_indices{i}, :, :) = whitened_eegs{i};
            end
            
            % Append the result to sorted_eeg_all
            sorted_eeg_all{ses} = whitened_eegs_re;
        end

        % Save the whitened EEG data
        save_dir = sprintf('output/sleemory_retrieval/whiten_eeg_matlab');
        if ~isfolder(save_dir)
            mkdir(save_dir);
        end

        save_dict.whitened_data = sorted_eeg_all;
        save_dict.imgs_all = imgs_sub;
        save(fullfile(save_dir, sprintf('whiten_test_eeg_sub-%03d.mat', sub)), '-struct', 'save_dict');
    end


function whitened_data = mvnn(all_epoched_data)
    % MVNN Compute the covariance matrices of the EEG data, average them 
    % across image conditions, and whiten the EEG data.
    %
    % Parameters
    % ----------
    % all_epoched_data : cell array of arrays of shape (rep,channel,time)
    %     Epoched EEG data.
    %
    % Returns
    % -------
    % whitened_data : cell array of arrays of shape (rep,channel,time)
    %     Whitened EEG data.

    % Initialize
    num_images = length(all_epoched_data);
    tot_sigma = cell(num_images, 1);

    % Compute the covariance matrices
    for i = 1:num_images
        data = all_epoched_data{i};
        [num_rep, num_ch, num_time] = size(data);
        sigma = zeros(num_ch, num_ch, num_time);
        
        % Compute covariance for each time point
        for t = 1:num_time
            temp_data = squeeze(data(:,:, t));
            sigma(:, :, t) = cov(temp_data);
        end
        
        % Average covariance matrices across time points
        tot_sigma{i} = mean(sigma, 3);
    end
    format long
    % Average the covariance matrices across image conditions
    mean_sigma = mean(cat(3, tot_sigma{:}), 3);
    mean_sigma = round(mean_sigma, 15);

    % Compute the inverse of the covariance matrix
    pyenv;
    np = py.importlib.import_module('numpy');
    linalg = py.importlib.import_module('scipy.linalg');

    sigma_inv = linalg.fractional_matrix_power(mean_sigma, -0.5);

    sigma_inv_data = np.ravel(sigma_inv);
    sigma_inv = double(py.array.array('d', sigma_inv_data));

    sigma_inv = reshape(sigma_inv, [58, 58]);

    % otherwise
    % sigma_inv = inv(real(sqrtm(mean_sigma)));
    % disp(sigma_inv);

    % Whiten the data
    whitened_data = cell(num_images, 1);
    for i = 1:num_images
       data = all_epoched_data{i};
       whiten = zeros(num_rep, num_ch, num_time);
       for t = 1: num_time
           temp_data = data(:,:, t);
           whiten(:,:, t) = temp_data * sigma_inv;
       whitened_data{i} = whiten;
       end
    end
end