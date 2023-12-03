function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = off4(data, labels, train_ratio, k)

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

[fs_index,times_k_off1] = off4Methods(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data(:,fs_index),labels,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data(:,fs_index),labels,train_ratio);

function [fs_index,times_k_off1] = off4Methods(feat, ~, num_features)
tic
% Perform TV
% Number of features
[num_data, dim] = size(feat);
% Start 
TV = zeros(1,dim);
for d = 1:dim
  % Mean of each feature
  mu    = mean(feat(:,d));
  % Term variance (7)
  TV(d) = (1 / num_data) * sum((feat(:,d) - mu) .^ 2);
end
times_k_off1 = toc;
% Larger value offer better information
[~, idx] = sort(TV,'descend');
% Select features based on selected positions 
fs_index       = idx(1:num_features);