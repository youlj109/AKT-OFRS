function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = Sstream1(data_all, labels_all, train_ratio, k)
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

[fs_index,times_k_off1] = OFS(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);

function [selected_index, timeUse] = OFS(data,~,B_range) 
tic
%set training and testing data
[n,d] = size(data);
epoch = ceil(2*d/n);
times = 1;  % run 10 times for calculating mean accuracy

% parameters
lambda = 1e-2; % regularization parameter
eta = 10^(-6.5);

nn1 = 1;

for p = 1: nn1
    B = B_range(p);
     
    errorNum = zeros(times,1);
    feaNum = zeros(times,epoch);
    allw = zeros(d-1, times);
    avgCumLoss =  zeros(times,epoch);
    
    for run = 1:times
        errNum = 0;
        w = zeros((d-1),1);
        loss = 0;
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = w' * x;
                loss = loss + max(0, 1 - y * pred_v) + lambda/2*sum(w.^2);
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                if y*pred_v < 1
                    w = (1 - lambda*eta)*w + eta*y*x;
                    l2norm = norm(w,2);
                    w = min(1, 1/(sqrt(lambda)*l2norm)) * w;
                    nnzNum = nnz(w);
                    if nnzNum > B
                        [rows,~, val_w] = find(abs(w));
                        [~,id] = sort(val_w,'descend');
                        w(rows(id(1:nnzNum-B))) = 0;
                    end
                else
                    w = (1 - lambda*eta)*w;
                end
            end
            
            feaNum(run,o) =  nnz(w);
            avgCumLoss(run,o) = loss/(o*n);
        end
        
        errorNum(run) = errNum;
        allw(:,run) = w;
    end    
end

%learning module
%example of Fast-OSFS for discrete data
[~,~, val_w] = find(abs(w));
[~,id] = sort(val_w,'descend');
selected_index = id(1:min(length(id),B_range));
timeUse = toc;