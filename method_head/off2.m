function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = off2(data, labels, train_ratio, k)

num_samples = size(data, 1);
num_features = floor(size(data, 2)*k);
num_trains = floor(num_samples * train_ratio);
trainData = data(1:num_trains, :);
trainLabels = labels(1:num_trains);
%初始化
avgF1_k_off1 = zeros([2,1]);
avgAUC_k_off1 = zeros([2,1]);
OA_k_off1 = zeros([2,1]);
AA_k_off1 = zeros([2,1]);

%特征选择

[fs_index,times_k_off1] = off2Methods(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data(:,fs_index),labels,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data(:,fs_index),labels,train_ratio);

function [fs_index,times_k_off1] = off2Methods(trainData, trainLabels, num_features)
tic
% Perform NCA
model    = fscnca(trainData,trainLabels);
times_k_off1 = toc;
% Weight
weight   = model.FeatureWeights; 
% Higher weight better features
[~, RANKED] = sort(weight,'descend');
fs_index = RANKED(1:num_features);