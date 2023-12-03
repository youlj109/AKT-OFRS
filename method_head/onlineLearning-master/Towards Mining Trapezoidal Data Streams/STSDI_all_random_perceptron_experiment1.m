clear;
load('a8a.mat');
C=10.^[-1 0 4 0 0 ];

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

result1=[];
result2=[];
mistake_number=[];   
for i=1:20
    temp=[];
    ID = ID_list(i,:);     
    %STSD
    [classifier_stsd, err_count_stsd, run_time_stsd, mistakes_stsd, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD(X, Y, NumFeature, ID, C(1,1) , lambda, 1);
     
    %STSD with no sparse.
    [classifier_psd, err_count_stsdf, run_time_psd, mistakes_psd, mistakes_idx_psd, SVs_psd, TMs_psd] = STSD_all(X, Y, ID, C(1,2),1);
         
    %STSD ramdonly select features
    [classifier_stsd, err_count_stsdr, run_time_stsd, mistakes_stsd, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_rand(X, Y, NumFeature, ID, C(1,3) , lambda, 1);
         
    %STSD with perceptron update strategy
    [classifier_stsd, err_count_stsdper, run_time_stsd, mistakes_stsd, mistakes_idx_stsd, SVs_stsd, TMs_stsd] = STSD_per(X, Y, NumFeature, ID, C(1,5) , lambda, 1);
         
    temp=[temp;err_count_stsd;err_count_stsdf;err_count_stsdr;err_count_stsdper];
    mistake_number=[mistake_number,temp];     
end

result1=[result1,mean(mistake_number,2)];
result2=[result2,std(mistake_number,0,2)];
result=[result1,result2];
