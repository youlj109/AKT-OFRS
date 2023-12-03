clear;
load('german.mat');
 
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
C1=10.^[-1];
C2=10.^[-4];
lambda = 0.001;

mistake_number=[];
t_mistakes_stsd=[];
t_mistakes_stsd1=[];
t_mistakes_stsd2=[];

%% run experiments:
for i=1:20
    temp=[];
    ID = ID_list(i,:);
    
    %STSD         
    [classifier_stsd, err_count_stsd, run_time_stsd, mistakes_stsd, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD(X, Y, NumFeature, ID, 0.1 , lambda, 0);
    t_mistakes_stsd=[t_mistakes_stsd;mistakes_stsd];  
    temp=[temp;err_count_stsd];          
    
    %STSD-I and STSD-II
    [classifier_stsd1, err_count_stsd1, run_time_stsd1, mistakes_stsd1, mistakes_idx_stsd1, SVs_stsd1, TMs_stsd1] = STSD(X, Y, NumFeature, ID, C1, lambda,1);
    temp=[temp;err_count_stsd1]; 
    t_mistakes_stsd1=[t_mistakes_stsd1;mistakes_stsd1];      
    
    [classifier_stsd2, err_count_stsd2, run_time_stsd2, mistakes_stsd2, mistakes_idx_stsd2, SVs_stsd2, TMs_stsd2] = STSD(X, Y, NumFeature, ID, C2, lambda,2);
    temp=[temp;err_count_stsd2];
    t_mistakes_stsd2=[t_mistakes_stsd2;mistakes_stsd2];
 
    mistake_number=[mistake_number,temp];
end

result1=mean(mistake_number,2);
result2=std(mistake_number,0,2);
result=[result1,result2]

mistakes_idx = 1:length(mistakes_idx_stsd);
%% print and plot results
figure
figure_FontSize=18;
mean_mistakes_stsd = mean(t_mistakes_stsd);
semilogx(2.^mistakes_idx, mean_mistakes_stsd,'k.-');
hold on
mean_mistakes_stsd1 = mean(t_mistakes_stsd1);
semilogx(2.^mistakes_idx, mean_mistakes_stsd1,'b-s');
mean_mistakes_stsd2 = mean(t_mistakes_stsd2);
semilogx(2.^mistakes_idx, mean_mistakes_stsd2,'g-x');
legend('STSD','STSD-I','STSD-II');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top','Position',[400  0.328 1]);
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle','Position',[0.25  0.52  1]);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca,'FontSize',12);
grid
