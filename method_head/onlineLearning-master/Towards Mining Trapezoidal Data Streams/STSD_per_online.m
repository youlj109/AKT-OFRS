function [w_t, err_count,time] = STSD_per_online(NumFeature, C,lambda,algorithm, dimension,dataset,w_t,instance)
%w_t=zeros(dimension,1);
err_count=[];
%NumFeature=options.NumFeature;
err=0;
t=0;
dimen=[0.1:0.1:1];
 fid = fopen(dataset, 'r');
     tic
     while ~feof(fid)
     a=fgetl(fid);
    data=str2num(a);

 
n  = length(data);
%ID_list = ID_ALL; % the sequence
Y = data(1,1); % label
X = reshape(data(1,2:n),[2 (n-1)/2]); % features


         
         t=t+1;
  

% ID = id_list;

%t_tick = options.t_tick;
%mistakes = [];
%mistakes_idx = [];
%SV = [];
%SVs = [];
%TMs=[];
%eta = 0.2;
% lambda = 0.01;
%k = 2;
%dim_ini=max(1,floor(dimen(1,1)*dimension));
%w_1=zeros(dim_ini,1);     % initialize the weight vector
%% loop
%tic
step=max(1,floor(1/length(dimen)*instance));
% w_t=w;
% flag=0;

%for t = 1:length(ID),
    s=max(floor((t-1)/step),1);
    
    %id = ID(t);

    %% prediction
    dimen_t=max(1,floor(dimen(1,s)*dimension));
%     if length(w_t)<dimen_t
%         flag=1;
%     end
    
    f_t=0;
    for j=1:size(X,2)
        if X(1,j)< dimen_t+1
            f_t=f_t+w_t(j,1)*X(2,j);
        else
            break;
        end
    end
    x_t=X(2,1:j-1);
%     x_t=X(id,1:dimen_t)';
%     if length(x_t)>length(w_t)
%         f_t=w_t'*x_t(1:length(w_t));
%         flag=1;
%     else
%         f_t=w_t'*x_t;
%     end
    y_t=Y;

    if y_t*f_t<=0,
       err=err+1;
    end 
    
    err_count=[err_count;err];
    
%     down=norm(x_t);
     if y_t*f_t<=1,
%         tao_t_PA=(1-y_t*f_t)/down;
%         tao_t_PAI=min(C,tao_t_PA);
%         tao_t_PAII=(1-y_t*f_t/(down+1/(2*C)));
%         
%         if algorithm==0
%             tao_t=tao_t_PA;
%         else if algorithm==1
%                 tao_t=tao_t_PAI;
%             else if algorithm==2
%                     tao_t=tao_t_PAII;
%                 end
%             end
%         end
        
        for j=1:length(x_t)
            index=X(1,j);
            w_t(index,1)=w_t(index,1)+y_t*X(2,j);
        end
 %       w_t= w_t+tao_t*y_t*x_t(1:length(w_t));
%         if flag==0;
            w_t = w_t *min(1,1/(sqrt(lambda)*sum(abs(w_t))));
            w_t=truncate_online(w_t,dimen_t,NumFeature);
%         end
%         SV = [SV id];
     end
        if t==instance
        break;
    end
     end
         time=toc;
     fclose(fid);
    
    
%     if flag==1;
%        w_new=tao_t*y_t*x_t(length(w_t)+1:end);
%        w_t=[w_t;w_new];
%        w_t = w_t *min(1,1/(sqrt(lambda)*sum(abs(w_t))));
%        w_t=truncate(w_t,NumFeature);
%        flag=0;
%     end

%    run_time = toc;
%      if (t==k)
%          k = 2*k;
%          mistakes = [mistakes err_count/t];
%          mistakes_idx = [mistakes_idx t];
%          SVs = [SVs length(SV)];
%          TMs=[TMs run_time];
%      end
end

% classifier.SV = SV;
% classifier.w_t = w_t;
% run_time = toc;

