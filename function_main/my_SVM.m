function [avgF1, avgAUC, OA, avgAA, times] = my_SVM(data, labels, trainRatio)
num_samples = size(data, 1);
num_trains = floor(num_samples * trainRatio);
trainData = data(1:num_trains, :);
trainLabels = labels(1:num_trains);
testData = data(num_trains+1:end, :);
testLabels = labels(num_trains+1:end);

% 训练 SVM 分类器
tic
svmModel = fitcecoc(trainData, trainLabels);

% 在测试集上进行预测
[predictedLabels, scores] = predict(svmModel, testData);
times = toc;

% 计算F1分数和AUC
classes = unique(labels);
counts = histcounts(labels, 'BinMethod', 'integers');
numClasses = length(classes);
f1Scores = zeros(numClasses, 1);
for i = 1:numClasses
    truePositive = sum(predictedLabels == classes(i) & testLabels == classes(i));
    falsePositive = sum(predictedLabels == classes(i) & testLabels ~= classes(i));
    falseNegative = sum(predictedLabels ~= classes(i) & testLabels == classes(i));

    % 处理除数为0的情况
    if truePositive == 0 && falsePositive == 0
        precision = 0; % 默认设置为0
    else
        precision = truePositive / (truePositive + falsePositive);
    end

    if truePositive == 0 && falseNegative == 0
        recall = 0; % 默认设置为0
    else
        recall = truePositive / (truePositive + falseNegative);
    end

    % 计算F1分数
    if precision == 0 && recall == 0
        f1Scores(i) = 0; % 当precision和recall都为0时，F1分数为0
    else
        f1Scores(i) = 2 * (precision * recall) / (precision + recall);
    end
end


avgF1 = sum(f1Scores .* (counts' / num_samples));

% 计算AUC
[~,index] = max(counts);%更关注数量最多的类
[~, ~, ~, AUC] = perfcurve(testLabels, scores(:, index), classes(index));

avgAUC = sum(AUC .* (counts' / num_samples));

% 计算 OA 和 AA
OA = sum(predictedLabels == testLabels) / numel(testLabels);

uniqueLabels = unique(labels);
numClasses = numel(uniqueLabels);
AA = zeros(numClasses, 1);
for i = 1:numClasses
    class = uniqueLabels(i);
    classIdx = testLabels == class;
    classCorrect = sum(predictedLabels(classIdx) == class);
    classSamples = sum(classIdx);
    AA(i) = classCorrect / classSamples;
end
avgAA = mean(AA);
end