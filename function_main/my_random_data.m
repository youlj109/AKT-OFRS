function [lables_after_random,data_after_random] = my_random_data(data,labels)
%先对行随机排列
% 创建一个分层抽样的交叉验证分区对象，将数据划分为训练集和测试集
c = cvpartition(labels, 'HoldOut', 0.2);

% 获取训练集的索引
trainIdx = training(c);

% 获取测试集的索引
testIdx = test(c);

% 根据索引从特征矩阵X和类标签向量y中获取训练集和测试集
X_train = data(trainIdx, :);
y_train = labels(trainIdx);
X_test = data(testIdx, :);
y_test = labels(testIdx);
%再对列随机排列
data_after_random = [X_train;X_test];
lables_after_random = [y_train;y_test];
