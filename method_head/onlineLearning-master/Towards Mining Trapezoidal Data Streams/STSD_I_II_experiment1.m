clear;
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
C=10.^[-4:1:8];
lambda = 0.001;

result1=[];
result2=[];
%% run experiments:
 for j=1:length(C);
     mistake_number=[];  
     for i=1:20
         temp=[];
         ID = ID_list(i,:); 
         
         %STSD
         [classifier_stsd0, err_count_stsd0, run_time_stsd0, mistakes_stsd0, mistakes_idx_stsd0, SVs_stsd0, TMs_stsd0] = STSD(X, Y, NumFeature, ID, C(1,j), lambda,0);
         temp=[temp;err_count_stsd0]; 
         %STSD-I and STSD-II
         [classifier_stsd1, err_count_stsd1, run_time_stsd1, mistakes_stsd1, mistakes_idx_stsd1, SVs_stsd1, TMs_stsd1] = STSD(X, Y, NumFeature, ID, C(1,j), lambda,1);
         temp=[temp;err_count_stsd1]; 
         [classifier_stsd2, err_count_stsd2, run_time_stsd2, mistakes_stsd2, mistakes_idx_stsd2, SVs_stsd2, TMs_stsd2] = STSD(X, Y, NumFeature, ID, C(1,j), lambda,2);
         temp=[temp;err_count_stsd2]; 
         
         mistake_number=[mistake_number,temp];     
     end
     
     result1=[result1,mean(mistake_number,2)];
     result2=[result2,std(mistake_number,0,2)];   
 end