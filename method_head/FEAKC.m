function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = FEAKC(data_all, labels_all, train_ratio, k)
data = data_all(1:floor(size(data_all,1)/2),:);%%%%%%%%%%%feature stream的样本数量是一半
labels = labels_all(1:floor(size(data_all,1)/2));
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

[fs_index,times_k_off1] = AKC_main(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);


function [maxIndexFeatures, timeUse] = AKC_main(data,labels,num_features)
tic

[n, m] = size(data); % 数据维度
numClasses = max(labels); % 类别数量

% 计算每个特征的每个类别的标准差和样本数量
classStd = zeros(m, numClasses);
classCount = zeros(1, numClasses);
featureStd = std(data);
featureCount = size(data, 1);

for i = 1:numClasses
    classData = data(labels == i, :);
    classStd(:, i) = std(classData);
    classCount(i) = size(classData, 1);
end

% 计算每个特征的指标
expandedFeatureStd = repmat(featureStd, numClasses, 1); % 扩展 featureStd 的维度
classWeights = classCount / featureCount;
%classStd(classStd==0)=1;
index = (expandedFeatureStd' > (classStd)) * classWeights';
%index = log(expandedFeatureStd' ./ (classStd)) * classWeights';
%[~,index] = fscmrmr(data, labels);
%[~,test1] = sort(index,'descend');
%test2 = test1(1:num_features);


% 生成特征的二进制向量
binaryVector = zeros([m,(numClasses^2-numClasses)/2]);
for j = 1:m
    for k1 = 1:numClasses-1
        for k2 = k1:numClasses
            if classStd(j, k1)>classStd(j, k2)
                binaryVector(j,k1) = 1;
            end
        end
    end
end

% 根据二进制向量各个维度的方差排序
variances = var(binaryVector);
[~, sortedIndices] = sort(variances, 'descend');
sortedBinaryVector = binaryVector(:, sortedIndices);
N = min(ceil(log2(num_features)),(numClasses^2-numClasses)/2);
% 二叉树聚类
numClusters = 2^N;
clusters = cell(1, numClusters);
clusters{1} = 1:m;

for i = 1:N
    numParentClusters = 2^(i-1);
    numChildClusters = 2^i;

    for j = 1:numParentClusters
        parentCluster = clusters{j};
        features = sortedBinaryVector(parentCluster, :);
        childLeft = parentCluster(features(:, i) == 0);
        childRight = parentCluster(features(:, i) == 1);
        clusters{j} = childLeft;
        clusters{numParentClusters+j} = childRight;
    end
end

% 在每个类中找到最大指标的特征
maxIndexFeatures = [];
count_clu =0;
for i_c = 1:length(clusters)
    if ~isempty(clusters{i_c})
        count_clu = count_clu+1;
    end
end

for j = 1:numClusters
    if isempty(clusters{j})
        continue
    end
    clusterFeatures = clusters{j};
    [~,index_max] = sort(index(clusterFeatures),'descend');

    maxIndexFeatures = [maxIndexFeatures, clusterFeatures(index_max(1:min(ceil(num_features/count_clu),length(clusters{j}))))];
end
timeUse = toc;