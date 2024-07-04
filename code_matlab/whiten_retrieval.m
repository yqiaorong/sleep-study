% List of image names
img_dir = 'dataset/sleemory_retrieval/image_set';
img_files = dir(fullfile(img_dir, '*.jpg'));
imgs_names = cellfun(@(x) x(1:end-4), {img_files.name}, 'UniformOutput', false);

    for sub = 2:5
        if sub == 17
            continue;
        end
        
        % Load the test EEG data
        eeg_dir = sprintf('dataset/sleemory_retrieval/preprocessed_data/');
        data = load(fullfile(eeg_dir, sprintf('sleemory_retrieval_dataset_sub-%03d.mat', sub)));
        eegs_sub = data.ERP_all; % (1, 2)
        imgs_sub = data.imgs_all; % (1, 2)
        clear data;

        sorted_eeg_all = {}; % Final data of two sessions
        for ses = 1:2
            eegs_ses = eegs_sub{1, ses}; % (num_trials, num_ch, num_time)
            imgs_ses = imgs_sub{1, ses}(:, 1);

            % Classify EEG data according to image names
            whitened_data_re = nan(size(eegs_ses)); % (num_trials, num_ch, num_time)
            for i = 1:length(imgs_names)
                name = imgs_names{i};
                mask = strcmp(imgs_ses, name);

                % Mark the index
                true_idx = find(mask);

                % Extract the EEG
                eeg = eegs_ses(mask, :, :); % (num_trials_per_img, num_ch, num_time)

                % Whiten the data
                whitened_data = mvnn({eeg});
                whitened_data = squeeze(whitened_data{1}); % (num_trials_per_img, num_ch, num_time)

                % Assign the whitened data to final whitened data with original order
                whitened_data_re(true_idx, :, :) = whitened_data;
                clear whitened_data;
            end

            % Append two sessions data
            sorted_eeg_all{ses} = whitened_data_re;
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

    % Average the covariance matrices across image conditions
    mean_sigma = mean(cat(3, tot_sigma{:}), 3);

    % Compute the inverse of the covariance matrix
    sigma_inv = inv(mean_sigma)^(0.5)

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