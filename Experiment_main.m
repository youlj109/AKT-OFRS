clc
clear
 % 创建并行池
% 读取数据集名称
folder = 'D:\study\research\code\code\data\'; % 替换为文件夹路径%%记得把code整个文件夹都加入工作空间
files = dir(fullfile(folder, '*.mat')); % 读取文件夹中所有的 .mat 文件
fileNames = {files.name}; % 获取文件名

datasets_name = cell(1, numel(fileNames)); % 初始化数据集名称数组
datas = cell(17,1);
labelss = cell(17,1);
for i_1 = 1:numel(fileNames)
    fprintf('数据集：%s\n', fileNames{i_1});
    datasets_name{i_1} = fullfile(folder, fileNames{i_1});
    load(datasets_name{i_1});
    datas{i_1} = (data);
    labelss{i_1} =(labels);
end

% 数据集数量
datasets_num = length(datasets_name);

% 训练集比例
train_ratio = 0.8;

% 方法数量
method_num = 19;

%随机的数量
rand_num = 1;

% k的数量
k_num = 10;

% 步骤数量
step_num = 1;

% 初始化结果存储变量
%knn
avgF1_alldata_knn = cell(datasets_num, 1);
avgAUC_alldata_knn = cell(datasets_num, 1);
OA_alldata_knn = cell(datasets_num, 1);
AA_alldata_knn = cell(datasets_num, 1);


%SVM
avgF1_alldata_SVM = cell(datasets_num, 1);
avgAUC_alldata_SVM = cell(datasets_num, 1);
OA_alldata_SVM = cell(datasets_num, 1);
AA_alldata_SVM = cell(datasets_num, 1);

times_alldata = cell(datasets_num, 1);

% 各数据集效果
%knn
selection_knn = zeros(datasets_num, 5);

avgF1_ero_all_knn = cell(datasets_num, 1);
avgAUC_ero_all_knn = cell(datasets_num, 1);
OA_ero_all_knn = cell(datasets_num, 1);
AA_ero_all_knn = cell(datasets_num, 1);
%SVM
selection_SVM = zeros(datasets_num, 5);

avgF1_ero_all_SVM = cell(datasets_num, 1);
avgAUC_ero_all_SVM = cell(datasets_num, 1);
OA_ero_all_SVM = cell(datasets_num, 1);
AA_ero_all_SVM = cell(datasets_num, 1);

times_ero_all = cell(datasets_num, 1);

data_succeed = [1,5,7,10,12,13,15,16];
for i_2 = 1:17 %每一个数据集
    if ismember(i_2,data_succeed)
        continue
    end
    warning('off')
    %读取数据
    data = datas{i_2};
    labels = labelss{i_2};

    %标签都用自然数表示
    labels = grp2idx(labels);

    %所有随机的结果(k*method_num*rand_num)
    %knn
    avgF1_all_knn = zeros([k_num,method_num,rand_num]);
    avgAUC_all_knn = zeros([k_num,method_num,rand_num]);
    OA_all_knn = zeros([k_num,method_num,rand_num]);
    AA_all_knn = zeros([k_num,method_num,rand_num]);

    %SVM
    avgF1_all_SVM = zeros([k_num,method_num,rand_num]);
    avgAUC_all_SVM = zeros([k_num,method_num,rand_num]);
    OA_all_SVM = zeros([k_num,method_num,rand_num]);
    AA_all_SVM = zeros([k_num,method_num,rand_num]);

    times_all = zeros([step_num,method_num,rand_num]);

    for random_times = 1:rand_num  %每一次随机
        [lables_after_random,data_after_random] = my_random_data(data,labels);
        if random_times == 1
            %Knn
            [avgF1_Knn, avgAUC_Knn, OA_Knn, AA_Knn, times_knn_eachrand] = my_Knn(data_after_random,lables_after_random ,train_ratio);
            %SVM
            [avgF1_SVM, avgAUC_SVM,OA_SVM, AA_SVM, times_SVM_eachrand] = my_SVM(data_after_random,lables_after_random ,train_ratio);
        end
    end
    AVG_KNN(i_2) = avgF1_Knn;
    AUC_KNN(i_2) = avgAUC_Knn;
    F1_SVM(i_2) = avgF1_SVM;
    AUC_SVM(i_2) = avgAUC_SVM;


%         %Knn
%         avgF1_eachrand_knn = zeros([k_num,method_num]);
%         avgAUC_eachrand_knn = zeros([k_num,method_num]);
%         OA_eachrand_knn = zeros([k_num,method_num]);
%         AA_eachrand_knn = zeros([k_num,method_num]);
% 
%         %SVM
%         avgF1_eachrand_SVM = zeros([k_num,method_num]);
%         avgAUC_eachrand_SVM = zeros([k_num,method_num]);
%         OA_eachrand_SVM = zeros([k_num,method_num]);
%         AA_eachrand_SVM = zeros([k_num,method_num]);
%         times_eachrand = zeros([step_num,method_num]);
% 
%         j_1 = 0;
%         for k_1 = 0.12:0.2/k_num:0.3 %每一个选择比例
%             k_1
%             j_1 = j_1+1;
%             %每一种方法：
%             % Offline
%             
%             [avgF1_k_off1, avgAUC_k_off1, OA_k_off1, AA_k_off1, times_k_off1] = off1(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_off2, avgAUC_k_off2, OA_k_off2, AA_k_off2, times_k_off2] = off2(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_off3, avgAUC_k_off3, OA_k_off3, AA_k_off3, times_k_off3] = off3(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_off4, avgAUC_k_off4, OA_k_off4, AA_k_off4, times_k_off4] = off4(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_off5, avgAUC_k_off5, OA_k_off5, AA_k_off5, times_k_off5] = off5(data_after_random, lables_after_random, train_ratio, k_1);
%             % feature_stream_other
%             [avgF1_k_feature_stream_other, avgAUC_k_feature_stream_other, OA_k_feature_stream_other, AA_k_feature_stream_other, times_k_feature_stream_other] = Fstream1(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_feature_stream_other2, avgAUC_k_feature_stream_other2, OA_k_feature_stream_other2, AA_k_feature_stream_other2, times_k_feature_stream_other2] = Fstream2(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_feature_stream_other3, avgAUC_k_feature_stream_other3, OA_k_feature_stream_other3, AA_k_feature_stream_other3, times_k_feature_stream_other3] = Fstream3(data_after_random, lables_after_random, train_ratio, k_1);
%             % sample_stream_other
%             [avgF1_k_sample_stream_other, avgAUC_k_sample_stream_other, OA_k_sample_stream_other, AA_k_sample_stream_other, times_k_sample_stream_other] = Sstream1(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_sample_stream_other2, avgAUC_k_sample_stream_other2, OA_k_sample_stream_other2, AA_k_sample_stream_other2, times_k_sample_stream_other2] = Sstream2(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_sample_stream_other3, avgAUC_k_sample_stream_other3, OA_k_sample_stream_other3, AA_k_sample_stream_other3, times_k_sample_stream_other3] = Sstream3(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_sample_stream_other4, avgAUC_k_sample_stream_other4, OA_k_sample_stream_other4, AA_k_sample_stream_other4, times_k_sample_stream_other4] = Sstream4(data_after_random, lables_after_random, train_ratio, k_1);
%             % trap_stream_other
%             [avgF1_k_trap_stream_other, avgAUC_k_trap_stream_other, OA_k_trap_stream_other, AA_k_trap_stream_other, times_k_trap_stream_other] = trap_stream_other(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_trap_stream_other2, avgAUC_k_trap_stream_other2, OA_k_trap_stream_other2, AA_k_trap_stream_other2, times_k_trap_stream_other2] = trap_stream_other2(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_trap_stream_other3, avgAUC_k_trap_stream_other3, OA_k_trap_stream_other3, AA_k_trap_stream_other3, times_k_trap_stream_other3] = trap_stream_other3(data_after_random, lables_after_random, train_ratio, k_1);
%             
%             %AKC
%             i_2
%             [avgF1_k_FEAKC, avgAUC_k_FEAKC,OA_k_FEAKC, AA_k_FEAKC, times_k_FEAKC] = FEAKC(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_SEAKC, avgAUC_k_SEAKC, OA_k_SEAKC, AA_k_SEAKC, times_k_SEAKC] = SEAKC(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_TEAKC, avgAUC_k_TEAKC, OA_k_TEAKC, AA_k_TEAKC, times_k_TEAKC] = TEAKC(data_after_random, lables_after_random, train_ratio, k_1);
%             [avgF1_k_AKC, avgAUC_k_AKC, OA_k_AKC, AA_k_AKC, times_k_AKC] = AKC(data_after_random, lables_after_random, train_ratio, k_1);
% 
%             %当前比例的avgF1,avgAUC_AKC,OA,AA,times
%             avgF1_k = [avgF1_k_off1, avgF1_k_off2, avgF1_k_off3, avgF1_k_off4, avgF1_k_off5,...
%                 avgF1_k_feature_stream_other,avgF1_k_feature_stream_other2, avgF1_k_feature_stream_other3...
%                 avgF1_k_sample_stream_other, avgF1_k_sample_stream_other2,avgF1_k_sample_stream_other3,avgF1_k_sample_stream_other4,...
%                 avgF1_k_trap_stream_other,avgF1_k_trap_stream_other2,avgF1_k_trap_stream_other3, ...
%                 avgF1_k_FEAKC, avgF1_k_SEAKC, avgF1_k_TEAKC, avgF1_k_AKC];
% 
%             avgAUC_k = [avgAUC_k_off1, avgAUC_k_off2, avgAUC_k_off3, avgAUC_k_off4, avgAUC_k_off5,...
%                 avgAUC_k_feature_stream_other,avgAUC_k_feature_stream_other2,avgAUC_k_feature_stream_other3, ...
%                 avgAUC_k_sample_stream_other,avgAUC_k_sample_stream_other2,avgAUC_k_sample_stream_other3,avgAUC_k_sample_stream_other4, ...
%                 avgAUC_k_trap_stream_other,avgAUC_k_trap_stream_other2,avgAUC_k_trap_stream_other3,...
%                 avgAUC_k_FEAKC, avgAUC_k_SEAKC, avgAUC_k_TEAKC, avgAUC_k_AKC];
% 
%             OA_k = [OA_k_off1,OA_k_off2,OA_k_off3,OA_k_off4,OA_k_off5,...
%                 OA_k_feature_stream_other,OA_k_feature_stream_other2,OA_k_feature_stream_other3,...
%                 OA_k_sample_stream_other,OA_k_sample_stream_other2,OA_k_sample_stream_other3,OA_k_sample_stream_other4,...
%                 OA_k_trap_stream_other,OA_k_trap_stream_other2,OA_k_trap_stream_other3,...
%                 OA_k_FEAKC,OA_k_SEAKC,OA_k_TEAKC,OA_k_AKC];
%             
%             AA_k = [AA_k_off1,AA_k_off2,AA_k_off3,AA_k_off4,AA_k_off5,...
%                 AA_k_feature_stream_other,AA_k_feature_stream_other2,AA_k_feature_stream_other3,...
%                 AA_k_sample_stream_other,AA_k_sample_stream_other2,AA_k_sample_stream_other3,AA_k_sample_stream_other4,...
%                 AA_k_trap_stream_other,AA_k_trap_stream_other2,AA_k_trap_stream_other3,...
%                 AA_k_FEAKC,AA_k_SEAKC,AA_k_TEAKC,AA_k_AKC];
% 
%             times_k = [times_k_off1,times_k_off2,times_k_off3,times_k_off4,times_k_off5,...
%                 times_k_feature_stream_other,times_k_feature_stream_other2,times_k_feature_stream_other3,...
%                 times_k_sample_stream_other,times_k_sample_stream_other2,times_k_sample_stream_other3,times_k_sample_stream_other4,...
%                 times_k_trap_stream_other,times_k_trap_stream_other2,times_k_trap_stream_other3,...
%                 times_k_FEAKC,times_k_SEAKC,times_k_TEAKC,times_k_AKC];
% 
%             
%             %Knn
%             avgF1_eachrand_knn(j_1,:) = avgF1_k(1,:);
%             avgAUC_eachrand_knn(j_1,:) = avgAUC_k(1,:);
%             OA_eachrand_knn(j_1,:) = OA_k(1,:);
%             AA_eachrand_knn(j_1,:) = AA_k(1,:);
% 
%             %SVM
%             avgF1_eachrand_SVM(j_1,:) = avgF1_k(2,:);
%             avgAUC_eachrand_SVM(j_1,:) = avgAUC_k(2,:);
%             OA_eachrand_SVM(j_1,:) = OA_k(2,:);
%             AA_eachrand_SVM(j_1,:) = AA_k(2,:);
% 
%             times_eachrand = times_eachrand + times_k;
%         end
% 
%         %当前随机所有比例的avgF1,avgAUC_AKC,OA,AA,times
%         %knn
%         avgF1_all_knn(:,:,random_times) = avgF1_eachrand_knn;
%         avgAUC_all_knn(:,:,random_times) = avgAUC_eachrand_knn;
%         OA_all_knn(:,:,random_times) = OA_eachrand_knn;
%         AA_all_knn(:,:,random_times) = AA_eachrand_knn;
% 
%         %SVM
%         avgF1_all_SVM(:,:,random_times) = avgF1_eachrand_SVM+avgF1_all_SVM;
%         avgAUC_all_SVM(:,:,random_times) = avgAUC_eachrand_SVM+avgAUC_all_SVM;
%         OA_all_SVM(:,:,random_times) = OA_eachrand_knn+OA_all_SVM;
%         AA_all_SVM(:,:,random_times) = AA_eachrand_knn+AA_all_SVM;
% 
%         times_all(:,:,random_times) = times_eachrand/10;
%     end
% 
%     %每个数据集的平均avgF1,avgAUC_AKC,OA,AA,times
%     %knn
%     avgF1_eachdata_knn = mean(avgF1_all_knn,3);
%     avgAUC_eachdata_knn = mean(avgAUC_all_knn,3);
%     OA_eachdata_knn = mean(OA_all_knn,3);
%     AA_eachdata_knn = mean(AA_all_knn,3);
% 
%     %SVM
%     avgF1_eachdata_SVM = mean(avgF1_all_SVM,3);
%     avgAUC_eachdata_SVM = mean(avgAUC_all_SVM,3);
%     OA_eachdata_SVM = mean(OA_all_SVM,3);
%     AA_eachdata_SVM = mean(AA_all_SVM,3);
% 
%     times_eachdata = mean(times_all,3);
% 
% 
%     %每个数据集的误差范围
%     %knn
%     avgF1_ero_knn = (max(avgF1_all_knn,3) - min(avgF1_all_knn,3))/2;
%     avgAUC_ero_knn = (max(avgAUC_all_knn,3) - min(avgAUC_all_knn,3))/2;
%     OA_ero_knn = (max(OA_all_knn,3) - min(OA_all_knn,3))/2;
%     AA_ero_knn = (max(AA_all_knn,3) - min(AA_all_knn,3))/2;
% 
%     %SVM
%     avgF1_ero_SVM = (max(avgF1_all_SVM,3) - min(avgF1_all_SVM,3))/2;
%     avgAUC_ero_SVM = (max(avgAUC_all_SVM,3) - min(avgAUC_all_SVM,3))/2;
%     OA_ero_SVM = (max(OA_all_SVM,3) - min(OA_all_SVM,3))/2;
%     AA_ero_SVM = (max(AA_all_SVM,3) - min(AA_all_SVM,3))/2;
% 
%     times_ero = (max(times_all,3) - min(times_all,3))/2;
% 
%     %储存所有数据集
%     %指标
%     %knn
%     avgF1_alldata_knn{i_2} = avgF1_eachdata_knn;
%     avgAUC_alldata_knn{i_2} = avgAUC_eachdata_knn;
%     OA_alldata_knn{i_2} = OA_eachdata_knn;
%     AA_alldata_knn{i_2} = AA_eachdata_knn;
% 
%     %SVM
%     avgF1_alldata_SVM{i_2} = avgF1_eachdata_SVM;
%     avgAUC_alldata_SVM{i_2} = avgAUC_eachdata_SVM;
%     OA_alldata_SVM{i_2} = OA_eachdata_SVM;
%     AA_alldata_SVM{i_2} = AA_eachdata_SVM;
% 
%     times_alldata{i_2} = times_eachdata;
%     %误差
%     %knn
%     avgF1_ero_all_knn{i_2} = avgF1_ero_knn;
%     avgAUC_ero_all_knn{i_2} = avgAUC_ero_knn;
%     OA_ero_all_knn{i_2} = OA_ero_knn;
%     AA_ero_all_knn{i_2} = AA_ero_knn;
% 
%     %SVM
%     avgF1_ero_all_SVM{i_2} = avgF1_ero_SVM;
%     avgAUC_ero_all_SVM{i_2} = avgAUC_ero_SVM;
%     OA_ero_all_SVM{i_2} = OA_ero_SVM;
%     AA_ero_all_SVM{i_2} = AA_ero_SVM;
% 
%     times_ero_all{i_2} = times_ero;
    %看所有数据集是否SOTA
    %knn
%     for rows = 1:k_num
%         if avgF1_eachdata_knn(rows,end)>avgF1_eachdata_knn(rows,1:6)
%             selection_knn(i,1) = selection_knn(i,1)+1;
%         end
%         if avgAUC_eachdata_knn(rows,end)>avgAUC_eachdata_knn(rows,1:6)
%             selection_knn(i,2) = selection_knn(i,2)+1;
%         end
%         if OA_eachdata_knn(rows,end)>OA_eachdata_knn(rows,1:6)
%             selection_knn(i,3) = selection_knn(i,3)+1;
%         end
%         if AA_eachdata_knn(rows,end)>AA_eachdata_knn(rows,1:6)
%             selection_knn(i,4) = selection_knn(i,4)+1;
%         end
%     end
% 
%     for rows = 1:step_num
%         if times_eachdata(rows,end)<times_eachdata(rows,1:6)
%             selection_knn(i,5) = selection_knn(i,5)+1;
%         end
%     end
% 
%      %SVM
%     for rows = 1:k_num
%         if avgF1_eachdata_SVM(rows,end)>avgF1_eachdata_SVM(rows,1:6)
%             selection_SVM(i,1) = selection_SVM(i,1)+1;
%         end
%         if avgAUC_eachdata_SVM(rows,end)>avgAUC_eachdata_SVM(rows,1:6)
%             selection_SVM(i,2) = selection_SVM(i,2)+1;
%         end
%         if OA_eachdata_SVM(rows,end)>OA_eachdata_SVM(rows,1:6)
%             selection_SVM(i,3) = selection_SVM(i,3)+1;
%         end
%         if AA_eachdata_SVM(rows,end)>AA_eachdata_SVM(rows,1:6)
%             selection_SVM(i,4) = selection_SVM(i,4)+1;
%         end
%     end
% 
%     for rows = 1:step_num
%         if times_eachdata(rows,end)<times_eachdata(rows,1:6)
%             selection_SVM(i,5) = selection_SVM(i,5)+1;
%         end
%     end
end

% %保存数据
% save('KNN_results.mat', 'avgF1_alldata_knn', 'avgAUC_alldata_knn','OA_alldata_knn','AA_alldata_knn', 'times_alldata',...
%     'avgF1_ero_all_knn','avgAUC_ero_all_knn','OA_ero_all_knn','AA_ero_all_knn','times_ero_all');
% 
% 
% save('SVM_results.mat', 'avgF1_alldata_SVM', 'avgAUC_alldata_SVM','OA_alldata_SVM','AA_alldata_SVM', 'times_alldata',...
%     'avgF1_ero_all_SVM','avgAUC_ero_all_SVM','OA_ero_all_SVM','AA_ero_all_SVM','times_ero_all');
% 
% save('ALLDATA_knn_results.mat','avgF1_Knn', 'avgAUC_Knn', 'OA_Knn', 'AA_Knn', 'times_knn_eachrand')
% 
% save('ALLDATA_SVM_results.mat','avgF1_SVM', 'avgAUC_SVM', 'OA_SVM', 'AA_SVM', 'times_SVM_eachrand')


