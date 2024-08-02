for sub = 2
    if sub == 17
        continue;
    end
    
    % Load the matlab whitened EEG data
    meeg_dir = sprintf('output/sleemory_retrieval/whiten_eeg_matlab/');
    data = load(fullfile(meeg_dir, sprintf('whiten_test_eeg_sub-%03d.mat', sub)));
    meegs_sub = data.whitened_data; % (2, 100, 58, 626)
    size(meegs_sub)
    mimgs_sub = data.imgs_all; % (1, 2)
    clear data;

    % Load the python whitened EEG data
    peeg_dir = sprintf('output/sleemory_retrieval/whiten_eeg_original/');
    data = load(fullfile(peeg_dir, sprintf('whiten_test_eeg_sub-%03d.mat', sub)));
    peegs_sub = data.whitened_data; % (2, 100, 58, 626)
    size(peegs_sub)
    pimgs_sub = data.imgs_all; 
    clear data;

    % Correlations
    corr_vals = zeros(2, 100, 626, 626);
    for ses = 1:2
       for sti = 1:100
           size(squeeze(meegs_sub(ses, sti, :, :)))
           size(squeeze(peegs_sub(ses, sti, :, :)))
           corr_vals(ses, sti,:,:) = corr(squeeze(meegs_sub(ses, sti, :, :)), squeeze(peegs_sub(ses, sti, :, :)));
       end
    end
end

% Display the correlation matrix as a 2D plot
mean_corr_vals = mean(corr_vals, 2);
size(mean_corr_vals)

figure;
imagesc(mean_corr_vals{1});
colorbar;
title('Correlation Matrix');
xlabel('Time (matlab)');
ylabel('Time (python)');
axis equal tight;


