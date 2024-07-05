num_feat = 1000; % input

% Load img names
imgDir = 'dataset/sleemory_retrieval/image_set';
imgFiles = dir(fullfile(imgDir, '*.jpg'));
imgNames = {imgFiles.name};
imgNames = cellfun(@(x) x(1:end-4), imgNames, 'UniformOutput', false);

layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'};
for sub = 2:16
    disp(sub)
    save_dir = sprintf('%s%03d', 'output/sleemory_retrieval/enc_acc/sub-', sub);
    if ~isfolder(save_dir)
       mkdir(save_dir);
    end

    % Load whitened EEG data
    test_path = sprintf('%s%03d%s', 'dataset/sleemory_retrieval/preprocessed_data/sleemory_retrieval_dataset_sub-', sub, '.mat');
    test_data = load(test_path);
    
    eegs_sub = test_data.ERP_all; % (1, 2)
    % Reshape the eegs_sub
    eegs(1,:,:,:) = eegs_sub{1};  % (2, 100, 58, 626)
    eegs(2,:,:,:) = eegs_sub{2};  % (2, 100, 58, 626)   
    imgs_sub = test_data.imgs_all; % (1, 2)
    imgs_sub = cat(2, imgs_sub{:})';  % (2, 100)
    for idx = 1:numel(layers)
        layer = layers{idx};
    
        % Load pred EEG data
        pred_path = sprintf('%s%d%s', 'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_', num_feat, 'feats.mat');
        pred_data = load(pred_path);
        pred_data = pred_data.(layer); % (4, 58, 363)
       
        % Duplicate the pred EEG data based on img names
        final_pred_data = zeros(2, 100, 58, 363);
        for ses = 1:2  
            for sti = 1:100
                switch imgs_sub{ses, sti}
                    case imgNames{1}
                        final_pred_data(ses, sti, :, :) = pred_data(1, :, :);
                    case imgNames{2}
                        final_pred_data(ses, sti, :, :) = pred_data(2, :, :);
                    case imgNames{3}
                        final_pred_data(ses, sti, :, :) = pred_data(3, :, :);
                    case imgNames{4}
                        final_pred_data(ses, sti, :, :) = pred_data(4, :, :);
                end
            end
        end
        size(final_pred_data); % (2, 100, 58, 363)
        
        % Correlations
        enc_acc = zeros(2, 100, 626, 363);
        for ses = 1:2
            for sti = 1:100
               enc_acc(ses, sti,:,:) = corr(squeeze(eegs(ses, sti, :, :)), squeeze(final_pred_data(ses, sti, :, :)));
            end
        end
    
        % Save data
        save_path = sprintf('%s%03d%s%s%s', 'output/sleemory_retrieval/enc_acc/sub-', sub, '/', layer', '_enc_acc_all_whitenFalse.mat');
        save(save_path, 'enc_acc');
           
    end
end
