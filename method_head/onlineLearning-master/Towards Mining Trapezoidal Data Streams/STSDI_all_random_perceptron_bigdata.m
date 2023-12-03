clear;
t_tick=100;
dataset='newKddb.t';
NumFeature=0.5;
C=10.^[0];
lambda = 0.001;
instance=20012498;
dimension=29890095;
w_0=ones(dimension,1);

err_count = 0;
Err_num_stsdI=[];
Err_num_stsdI_all=[];
Err_num_stsdI_rand=[];
Err_num_stsdI_perce=[];
Time_final=[];

 for j=1:length(NumFeature);
     runtime=[];    
     [w_t_stsdI, err_count_stsdI,time_stsdI] = STSD_online(NumFeature(1,j), C,lambda,1, dimension,dataset,w_0,instance);
     [w_t_stsdI_all, err_count_stsdI_all,time_stsdI_all] = STSD_all_online(NumFeature(1,j), C,lambda,1, dimension,dataset,w_0,instance);
     save all;
 end
max(Err_num_stsdI);

