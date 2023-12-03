clear;

C=10.^[-1 1 2 4];
load('a8a.mat');
 
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

NumFeature=0.5;
lambda = 0.001;

mistake_number=[];
t_mistakes_stsd1=[];
t_mistakes_stsd1_random=[];
t_mistakes_stsd1_all=[];
t_mistakes_stsd1_perceptron=[];

%% run experiments:
for i=1:20
    temp=[];
    ID = ID_list(i,:); 
    %STSD-I and STSD-II
    [classifier_stsd1, err_count_stsd1, run_time_stsd1, mistakes_stsd1, mistakes_idx_stsd1, SVs_stsd1, TMs_stsd1] = STSD(X, Y, NumFeature, ID, C(1,1), lambda,1);
    temp=[temp;err_count_stsd1]; 
    t_mistakes_stsd1=[t_mistakes_stsd1;mistakes_stsd1];

    %STSD with no sparse.
    [classifier_psd, err_count_stsdf, run_time_psd, mistakes_stsd1_all, mistakes_idx_psd, SVs_psd, TMs_psd] = STSD_all(X, Y, ID, C(1,2),1);
    t_mistakes_stsd1_all=[t_mistakes_stsd1;mistakes_stsd1_all];
    %STSDI with randomly selected features.
    [classifier_stsd, err_count_stsdr, run_time_stsd, mistakes_stsd1_random, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_rand(X, Y, NumFeature, ID, C(1,3) , lambda, 1);
    t_mistakes_stsd1_random=[t_mistakes_stsd1;mistakes_stsd1_random];
    %STSD with perceptron update
    [classifier_stsd, err_count_stsdper, run_time_stsd, mistakes_stsd1_perceptron, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_per(X, Y, NumFeature, ID, C(1,4) , lambda, 1);
    t_mistakes_stsd1_perceptron=[t_mistakes_stsd1;mistakes_stsd1_perceptron];  
    mistake_number=[mistake_number,temp];
end
result1=mean(mistake_number,2);
result2=std(mistake_number,0,2);
result=[result1,result2]
mistakes_idx = 1:length(mistakes_idx_stsd);
%% print and plot results
figure
figure_FontSize=18;

mean_mistakes_stsd1_random = mean(t_mistakes_stsd1_random);
semilogx(2.^mistakes_idx, mean_mistakes_stsd1_random,'b-s');
hold on
mean_mistakes_stsd1_all = mean(t_mistakes_stsd1_all);
semilogx(2.^mistakes_idx, mean_mistakes_stsd1_all,'g-x');
mean_mistakes_stsd1_perceptron = mean(t_mistakes_stsd1_perceptron);
semilogx(2.^mistakes_idx, mean_mistakes_stsd1_perceptron,'k.-');
mean_mistakes_stsd1 = mean(t_mistakes_stsd1);
semilogx(2.^mistakes_idx, mean_mistakes_stsd1,'r-*');
legend('STSDI-all','STSDI-rand','STSDI-per','STSDI');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top','Position',[400 0.376 1]);
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle','Position',[0.25 0.57  1]);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca,'FontSize',12);   
grid
