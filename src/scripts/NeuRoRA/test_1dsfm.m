%%Evaluation code for rotation averaging on real datasets

clc
clear

datasetPath = '../../../data/rotreal/processed';
resultPath = '../../../results/1dsfm';

%% Get list of scenes to evaluate
fnames = dir(fullfile(resultPath, '*.mat'));
scenes = {};
for i = 1 : length(fnames)
    [~, name, ~] = fileparts(fnames(i).name);
    scenes{end + 1} = name;
end


%%
mean_all = [];
median_all = [];
rmse_all = [];

for iScene = 1 : length(scenes)
    scene = scenes{iScene};
    disp(scene)
    
    fprintf('[%s]\n', scene)
    
    %%
    % Loads the datafile. It contains the following variables
    % - Rgt: Groundtruth absolute poses (3,3,N). This transforms points in
    %        the world frame to the camera frame.
    % - RR: Measured relative poses (3,3,N). Transforms point sin camera 
    %       Unused for evaluation j to camera i
    % - I: Edge indices (1-based). Each column contains indices (i, j).
    %      Used for evaluation
    % It then shows the statistics of the input raw data, i.e. the accuracy
    % of the measured relative transforms.
    clear {'Rgt', 'RR', 'I'}
    load(fullfile(datasetPath, [scene, '.mat']))
    fprintf('Validating Input data w.r.t Ground truth\n');
%     ValidateSO3Graph(Rgt,RR,I);close all;  % Ensure this is small
    
    %% Load the predicted poses
    load(fullfile(resultPath, [scene, '.mat']))
    
    %% Evaluate    
    fprintf('Comparing Estimated Rotations to Ground truth\n');
    [Ebest, ~, ~] = CompareRotationGraph_neurora(Rgt,R_pred);
    fprintf('\n')
    
    mean_all = [mean_all; Ebest(1)];
    median_all = [median_all; Ebest(2)];
    rmse_all = [rmse_all; Ebest(3)];
    
    fprintf('\n');
    
end

fprintf('Average errors: %.3f (mean), %.3f (median) % .3f (rmse)\n', mean(mean_all), mean(median_all), mean(rmse_all))