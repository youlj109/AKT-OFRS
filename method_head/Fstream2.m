function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = Fstream2(data_all, labels_all, train_ratio, k)
data = data_all(1:floor(size(data_all,1)/2),:);%%%%%%%%%%%feature stream的样本数量是一半
labels = labels_all(1:floor(size(data_all,1)/2),:);
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

[fs_index,times_k_off1] = FeatureStream(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);

function [selected_index, timeUse] = FeatureStream(fea,labels,m) 
%set training and testing data

%learning module
%example of Fast-OSFS for discrete data
[f,timeUse] = Alpha_Investing(fea, labels);
selected_index = f(1:min(length(f),m));