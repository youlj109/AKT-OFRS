function [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = Sstream2(data_all, labels_all, train_ratio, k)
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

[fs_index,times_k_off1] = SOFS(trainData, trainLabels, num_features);

%分类
%knn
[avgF1_k_off1(1), avgAUC_k_off1(1), OA_k_off1(1), AA_k_off1(1), ~] = my_Knn(data_all(:,fs_index),labels_all,train_ratio);

%SVM
[avgF1_k_off1(2), avgAUC_k_off1(2), OA_k_off1(2), AA_k_off1(2), ~] = my_SVM(data_all(:,fs_index),labels_all,train_ratio);

function [selected_index, timeUse] = SOFS(data,~,B_range) 
tic
[n,d] = size(data);
epoch = ceil(2*d/n);
times = 11;  % run 10 times for calculating mean accuracy

% parameters
gamma = 1e+05;
nn1 = 1;


for q = 1:nn1
    
    B = B_range(q);
    errorNum = zeros(times,1);
    feaNum = zeros(times,epoch); % feaNum
    meanLoss = zeros(times,epoch); % mean loss
    allw = zeros(d-1, times);
    
    
    for run = 1:times
        errNum = 0;
        mu = zeros((d-1),1);
        sigma = ones((d-1),1);
        loss = 0;
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = mu'* x;
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                loss = loss + (max(0, 1 - y*pred_v))^2;
                if y*pred_v < 1
                    belta = 1 / ((x.^2)'*sigma + gamma);
                    alpha = max(0, 1 - y*pred_v)*belta;
                    mu = mu + alpha * y * (sigma .* x);
                    sigma = 1./ ( 1./sigma + 1/gamma * (x.^2));
                    
                    % truncation operation
                    if size(sigma,1) > B
                        [val_sort,id] = sort(sigma,'descend');
                        mu(id(1: size(val_sort,1) - B )) = 0;
                    end
                    
                end
            end
            
            meanLoss(run,o) = loss/(o*n);
            feaNum(run,o) = nnz(mu);
        end
        
        errorNum(run) = errNum;
        allw(:,run) = mu;
    end
    
end

%learning module
%example of Fast-OSFS for discrete data
[~,id] = sort(sigma,'descend');
selected_index = id(1:min(length(id),B_range));
timeUse = toc;