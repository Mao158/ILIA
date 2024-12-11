clc;
clear;

addpath(genpath('dataset'));
addpath(genpath('evaluation'));
addpath(genpath('help_function'));

load('CAL500.mat'); data_str = 'CAL500';%502,68,174
% load('emotions.mat'); data_str = 'emotions';%593,72,6
% load('medical.mat'); data_str = 'medical';%978,1449,45
% load('image.mat'); data_str = 'image';%2000,294,5
% load('scene.mat'); data_str = 'scene';%2407,294,6
% load('arts.mat'); data_str = 'arts';%5000,462,26
% load('corel5k.mat'); data_str = 'corel5k';%5000,499,374
% load('education.mat'); data_str = 'education';%5000,550,33
% load('health.mat'); data_str = 'health';%8116,1483,32
% load('entertainment.mat'); data_str = 'entertainment';%8166,545,21

% data = PCA(data);
% data = zscore(data);
[num_data, num_dim] = size(data);
num_label = size(target,1);
para.num_fold = 10; % number of fold
para.data_str = data_str;
para.max_iter = 100;
para.tolerance = 0.1;
para.k = 20; % number of nearest neighbors
para.kernel_type = 'Poly'; % 'Linear','RBF','Poly','Sigmoid','Lapla'
para.kernel_para1 = 0.1; % 'Auto' or numerical value
para.kernel_para2 = 10; % Applicable to 'Poly' and 'Sigmoid'
para.kernel_para3 = 2; % Only applicable to 'Poly'
% trade-off parameters
para.mu = 0.001;
para.eta = 0.01;
para.gamma = 0.01;
% parameter of BRKNN
para.num_BRKNN_neighbour = 10;
% Parameters of MLKNN
para.num_MLKNN_neighbour = 10;
para.smooth = 1.0;

num_fold = para.num_fold;

Result_ILIA_BRKNN = zeros(num_fold, 6);
Result_ILIA_MLKNN = zeros(num_fold, 6);
TrainTime = zeros(num_fold,1);
Result_Metric = cell(num_fold, 1);
Result_Coef = cell(num_fold, 1);
Result_delta_M = zeros(num_fold, para.max_iter);

% Set a random seed to make the experiment reproducible
seed = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(seed);
% rng(1);
indices = crossvalind('Kfold',num_data,10);
for fold = 1 : num_fold
    seed2 = RandStream('mt19937ar','Seed',1);
    RandStream.setGlobalStream(seed2);

    test_logical = (indices == fold);
    train_logical = ~ test_logical;
    train_data = data(train_logical,:);
    test_data = data(test_logical,:);
    train_target = target(:,train_logical);
    test_target = target(:,test_logical);

    %     num_train = size(train_data,1);
    %     sum_class = sum(train_target,2); % Determine how many positive instances in each label
    %     condition = (sum_class >= 2) & (sum_class <= num_train - 2);  % when encountering severe class-imbalance problem, we ignore the corresponding label.
    %     train_target = train_target(condition,:);
    %     test_target = test_target(condition,:);

    % Label Importance Recovery
    F = LIR(train_data, train_target, para);

    % Centralize X
    mean_data = mean(train_data);
    train_data_centered = train_data - mean_data;
    test_data_centered = test_data - mean_data; % Highlight
    % Centralize F
    mean_F = mean(F);
    F_centered = F - mean_F;

    % Metric Learning
    tic;
    [M, Theta, Record_delta_M] = Metric_Learning(train_data_centered, F_centered, para, fold);

    % Saving convergence information
    Result_delta_M(fold, :) = Record_delta_M;

    % Saving training time
    TrainTime(fold,1) = toc;

    % saving learned metrics and model coefficients
    Result_Metric{fold} = M;
    Result_Coef{fold} = Theta;

    % BRKNN is coupled with ILIA: ILIA-BRKNN
    [Outputs_BRKNN, Pre_Labels_BRKNN] = ILIA_BRKNN_predict(train_data_centered, train_target, test_data_centered, para, M, Theta);
    [HammingLoss_BRKNN, RankingLoss_BRKNN, Coverage_BRKNN, Average_Precision_BRKNN, MacroF1_BRKNN, MacroAUC_BRKNN] = MLEvaluate(Outputs_BRKNN, Pre_Labels_BRKNN, test_target);
    Result_ILIA_BRKNN(fold,:) = [HammingLoss_BRKNN, RankingLoss_BRKNN, Coverage_BRKNN, Average_Precision_BRKNN, MacroF1_BRKNN, MacroAUC_BRKNN];

    % MLKNN is coupled with ILIA: ILIA-MLKNN
    [Prior, PriorN, Cond, CondN] = ILIA_MLKNN_train(train_data_centered, train_target, para, M, Theta);
    [Outputs_MLKNN, Pre_Labels_MLKNN] = ILIA_MLKNN_predict(test_data_centered, train_data_centered, train_target, Prior, PriorN, Cond, CondN, para, M, Theta);
    [HammingLoss_MLKNN, RankingLoss_MLKNN, Coverage_MLKNN, Average_Precision_MLKNN, MacroF1_MLKNN, MacroAUC_MLKNN] = MLEvaluate(Outputs_MLKNN, Pre_Labels_MLKNN, test_target);
    Result_ILIA_MLKNN(fold,:) = [HammingLoss_MLKNN, RankingLoss_MLKNN, Coverage_MLKNN, Average_Precision_MLKNN, MacroF1_MLKNN, MacroAUC_MLKNN];
end
Result_ILIA_BRKNN_mean = round(mean(Result_ILIA_BRKNN,1),3);
Result_ILIA_BRKNN_std = round(std(Result_ILIA_BRKNN,0,1),3);
Result_ILIA_MLKNN_mean = round(mean(Result_ILIA_MLKNN,1),3);
Result_ILIA_MLKNN_std = round(std(Result_ILIA_MLKNN,0,1),3);
TrainTime_mean = round(mean(TrainTime,1),3);
TrainTime_std = round(std(TrainTime,0,1),3);

% Print results of BRKNN-ILIA
fprintf('BRKNN-ILIA results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_ILIA_BRKNN_mean(1), Result_ILIA_BRKNN_std(1), Result_ILIA_BRKNN_mean(2), Result_ILIA_BRKNN_std(2), ...
    Result_ILIA_BRKNN_mean(3), Result_ILIA_BRKNN_std(3), Result_ILIA_BRKNN_mean(4), Result_ILIA_BRKNN_std(4), Result_ILIA_BRKNN_mean(5), Result_ILIA_BRKNN_std(5), Result_ILIA_BRKNN_mean(6), Result_ILIA_BRKNN_std(6));

% Print results of MLKNN-ILIA
fprintf('MLKNN-ILIA results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_ILIA_MLKNN_mean(1), Result_ILIA_MLKNN_std(1), Result_ILIA_MLKNN_mean(2), Result_ILIA_MLKNN_std(2), ...
    Result_ILIA_MLKNN_mean(3), Result_ILIA_MLKNN_std(3), Result_ILIA_MLKNN_mean(4), Result_ILIA_MLKNN_std(4), Result_ILIA_MLKNN_mean(5), Result_ILIA_MLKNN_std(5), Result_ILIA_MLKNN_mean(6), Result_ILIA_MLKNN_std(6));

% Print the training time
fprintf('The training time is %5.3f±%5.3f seconds:\n', TrainTime_mean, TrainTime_std);

