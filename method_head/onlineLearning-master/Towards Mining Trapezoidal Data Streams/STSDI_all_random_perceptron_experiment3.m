clear;
load('D:\study\online feature selection\OFS_code\OFS_full information\data\german.mat');
% load('D:\study\online feature selection\OFS_code\OFS_full information\data\a8a.mat');
NumFeature=2.^[2:1:6]*0.01;

C=10.^[-2 -2 -4 -2];
lambda = 0.001;

[n,d]  = size(data);
ID_list = ID_ALL; % the sequence
Y = data(1:n,1); % label
X = data(1:n,2:d); % features

stdX=std(X); % standard deviation
idx1=stdX~=0;
centrX=X-repmat(mean(X),size(X,1),1);
X(:,idx1)=centrX(:,idx1)./repmat(stdX(:,idx1),size(X,1),1);
X=(X-repmat(mean(X),size(X,1),1))./repmat(std(X),size(X,1),1);
X=X./repmat(sqrt(sum(X.*X,2)),1, size(X,2));
data=[Y,X];
t_tick=round(n/15);

t_mistakes_stsd=[];
t_mistakes_stsd1=[];
t_mistakes_stsd2=[];

result=[];
%% run experiments:
for j=1:length(NumFeature)
    mistake_number=[];
for i=1:20
    temp=[];
    ID = ID_list(i,:);
    [classifier_stsd1, err_count_stsd1, run_time_stsd1, mistakes_stsd1, mistakes_idx_stsd1, SVs_stsd1, TMs_stsd1] = STSD(X, Y, NumFeature(1,j), ID, C(1,1), lambda,1);
    temp=[temp;err_count_stsd1]; 
    t_mistakes_stsd1=[t_mistakes_stsd1;mistakes_stsd1];
    %STSD with no sparse.
    [classifier_psd, err_count_stsd1_all, run_time_psd, mistakes_stsd1_all, mistakes_idx_psd, SVs_psd, TMs_psd] = STSD_all(X, Y, ID, C(1,2),1);
    t_mistakes_stsd1_all=[t_mistakes_stsd1;mistakes_stsd1_all];  
    
    %STSDI with randomly selected features.
    [classifier_stsd, err_count_stsd1_random, run_time_stsd, mistakes_stsd1_random, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_rand(X, Y, NumFeature(1,j), ID, C(1,3) , lambda, 1);
    temp=[temp;err_count_stsd1_random]; 
    t_mistakes_stsd1_random=[t_mistakes_stsd1;mistakes_stsd1_random];        
    %STSD with perceptron update
    [classifier_stsd, err_count_stsd1_perceptron, run_time_stsd, mistakes_stsd1_perceptron, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_per(X, Y, NumFeature(1,j), ID, C(1,4) , lambda, 1);
    temp=[temp;err_count_stsd1_perceptron]; 
    t_mistakes_stsd1_perceptron=[t_mistakes_stsd1;mistakes_stsd1_perceptron];  
    temp=[temp;err_count_stsd1_all]; 
    
    mistake_number=[mistake_number,temp];
end
result1=1-mean(mistake_number,2)/n;
result=[result,result1];
end

result(4,1:4)=0;
figure
figure_FontSize=18;
bar(result')
hleg=legend('STSDI','STSDI-rand','STSDI-per','STSDI-all');
set(hleg, 'Position', [.235,.685,.1,.2]);
xlabel('Fraction of Selected Features');
ylabel('Classification Accuracy')
set(gca,'XTickLabel',{'0.04','0.08','0.16','0.32','0.64'},'FontSize',14);
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top','position',[3.0,-0.05,0.2]);
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle','position',[-0.65,0.4,0.2]);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
grid

