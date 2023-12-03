function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = Sstream3(data_all, labels_all, train_ratio, k)
data = data_all(:,1:floor(size(data_all,2)/2));%%%%%%%%%%%Sample stream的特征数量是一半%%%%%%%%
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

[fs_index,times_k_off1] = B_ARDA(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);

function [selected_index, timeUse] = B_ARDA(data,~,B_range) 
tic
%set training and testing data
[n,d] = size(data);
epoch = ceil(2*d/n); 
times = 1;  % run 10 times for calculating mean accuracy
% parameters
delta = 10.^(-2);
lambda = 10.^(-2);
eta  = 10.^(-4);

nn2 = 1;

for q = 1:nn2
    B = B_range(q);
    
    errorNum = zeros(times,1);
    feaNum = zeros(times,epoch); % feaNum
    meanLoss = zeros(times,epoch); % mean loss
    allw = zeros(d-1, times);
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
   
    
    for run = 1:times
        errNum = 0;
        w = zeros((d-1),1);
        loss = 0;
        s_t = zeros((d-1),1);
        H_t = zeros((d-1),1);
        sum_g_t = zeros((d-1),1);
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = w' * x;
                loss = loss + (max(0, 1 - y * pred_v))^2 + lambda/2 * sum(w.^2);
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                if y*pred_v < 1
                    g_t = - y * x * 2 *(1 - y*pred_v);
                    sum_g_t = sum_g_t + g_t;
                    s_t = sqrt( s_t.^2 + g_t.^2);
                    H_t = delta + s_t;
                else
                    g_t = zeros(d-1,1);
                end
                
                w = -eta * ( 1./ (lambda*eta*t + H_t) .* sum_g_t);
                
                % truncation operation
                if nnz(w) > B
                    [rows, ~, val_w] = find((w.^2).*H_t);
                    [val_sort,id] = sort(val_w,'ascend');
                    w(rows(id(1: size(val_sort,1) - B ))) = 0;
                end
            end
            
            meanLoss(run,o) = loss/(o*n);
            feaNum(run,o) = nnz(w);
        end
        
        errorNum(run) = errNum;
        allw(:,run) = w;
    end
end
%learning module
%example of Fast-OSFS for discrete data
[~, ~, val_w] = find((w.^2).*H_t);
[~,id] = sort(val_w,'descend');
selected_index = id(1:min(length(id),B_range));
timeUse = toc;