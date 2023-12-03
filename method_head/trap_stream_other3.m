function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = trap_stream_other2(data_all, labels_all, train_ratio, k)
data = data_all;
labels = labels_all;
num_samples = size(data, 1);
num_features = floor(size(data_all, 2)*k);
num_trains = floor(num_samples * train_ratio);
trainData = data(1:num_trains, :);
trainLabels = labels(1:num_trains);
%初始化
avgF1_k_off1 = zeros([2,1]);
avgAUC_k_off1 = zeros([2,1]);
OA_k_off1 = zeros([2,1]);
AA_k_off1 = zeros([2,1]);

%特征选择

[fs_index,times_k_off1] = TrapezoidalStream(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);

function [selected_index, timeUse] = TrapezoidalStream(fea,Y_labels,m) 
[n,~]  = size(fea);
ID_list = 1:n; % the sequence
Y = Y_labels; % label
X = fea; % features


stdX=std(X); % standard deviation
idx1=stdX~=0;
centrX=X-repmat(mean(X),size(X,1),1);
X(:,idx1)=centrX(:,idx1)./(repmat(stdX(:,idx1),size(X,1),1)+0.000000000001);

X=(X-repmat(mean(X),size(X,1),1))./(repmat(std(X),size(X,1),1)+0.000000000001);
X=X./repmat(sqrt(sum(X.*X,2)),1, size(X,2));


NumFeature=0.5;
C1=10.^[-1];
C2=10.^[-4];
lambda = 0.001;



% run experiments:
for i=1:1
    ID = ID_list(i,:);
    
    %STSD         
    [classifier_stsd, ~, timeUse, ~, ~, ~, ~] = STSD(X, Y, NumFeature, ID, 0.1 , lambda, 2);
end
[~,selected_index_all] = sort(abs(classifier_stsd.w_t),'descend');
selected_index = selected_index_all(1:m);