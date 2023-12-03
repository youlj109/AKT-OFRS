function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = STSD_rand(X, Y, NumFeature, id_list, C,lambda,algorithm)
dimen=[0.1:0.1:1];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SV = [];
SVs = [];
TMs=[];
eta = 0.2;
k = 2;
dim_ini=max(1,floor(dimen(1,1)*size(X,2)));
w_1=zeros(dim_ini,1);     % initialize the weight vector

tic
step=max(1,floor(1/length(dimen)*length(ID)));
w_t=w_1;
flag=0;

for t = 1:length(ID),
    j=max(floor((t-1)/step),1);
    id = ID(t);

    %% prediction
    dimen_t=max(1,floor(dimen(1,j)*size(X,2)));
    x_t=X(id,1:dimen_t)';
    if length(x_t)>length(w_t)
        f_t=w_t'*x_t(1:length(w_t));
        flag=1;
    else
        f_t=w_t'*x_t;
    end
    y_t=Y(id);

    if y_t*f_t<=0,
       err_count=err_count+1;
    end 
    
    if y_t*f_t<=1,
        tao_t_PA=(1-y_t*f_t)/norm(x_t);
        tao_t_PAI=min(C,tao_t_PA);
        tao_t_PAII=(1-y_t*f_t/(norm(x_t)+1/(2*C)));
        
        if algorithm==0
            tao_t=tao_t_PA;
        else if algorithm==1
                tao_t=tao_t_PAI;
            else if algorithm==2
                    tao_t=tao_t_PAII;
                end
            end
        end
        
        w_t= w_t+tao_t*y_t*x_t(1:length(w_t));
        if flag==0;
            %% random selection of features
            B=round(NumFeature*length(w_t));
            v_idx=zeros(size(w_t,1),1);
            perm_t=randperm(size(w_t,1));
            c_t=perm_t(1:B);
            v_idx(c_t)=1;
            w_t=w_t.*v_idx;
        end
        SV = [SV id];
    end
    
    if flag==1;
       w_new=tao_t*y_t*x_t(length(w_t)+1:end);
       w_t=[w_t;w_new];
       
       B=round(NumFeature*length(w_t));
       v_idx=zeros(size(w_t,1),1);
       perm_t=randperm(size(w_t,1));
       c_t=perm_t(1:B);
       v_idx(c_t)=1;
       w_t=w_t.*v_idx;
       flag=0;
    end

    run_time = toc;
    if (t==k)
        k = 2*k;
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end

classifier.SV = SV;
classifier.w_t = w_t;
run_time = toc;

