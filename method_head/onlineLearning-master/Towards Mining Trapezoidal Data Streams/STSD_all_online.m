function [w_t, err_count,time] = STSD_all_online(NumFeature, C,lambda,algorithm, dimension,dataset,w_t,instance)
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

Y = data(1,1); % label
X = reshape(data(1,2:n),[2 (n-1)/2]); % features
        
         t=t+1;

step=max(1,floor(1/length(dimen)*instance));

    s=max(floor((t-1)/step),1);
    

    dimen_t=max(1,floor(dimen(1,s)*dimension));

    
    f_t=0;
    for j=1:size(X,2)
        if X(1,j)< dimen_t+1
            f_t=f_t+w_t(j,1)*X(2,j);
        else
            break;
        end
    end
    x_t=X(2,1:j-1);

    y_t=Y;

    if y_t*f_t<=0,
       err=err+1;
    end 
    
    err_count=[err_count;err];
    
    down=norm(x_t);
    if y_t*f_t<=1,
        tao_t_PA=(1-y_t*f_t)/down;
        tao_t_PAI=min(C,tao_t_PA);
        tao_t_PAII=(1-y_t*f_t/(down+1/(2*C)));
        
        if algorithm==0
            tao_t=tao_t_PA;
        else if algorithm==1
                tao_t=tao_t_PAI;
            else if algorithm==2
                    tao_t=tao_t_PAII;
                end
            end
        end
        
        for j=1:length(x_t)
            index=X(1,j);
            w_t(index,1)=w_t(index,1)+tao_t*y_t*X(2,j);
        end

%            w_t = w_t *min(1,1/(sqrt(lambda)*sum(abs(w_t))));
%            w_t=truncate_online(w_t,dimen_t,NumFeature);

    end
        if t==instance
        break;
    end
     end
         time=toc;
     fclose(fid);

end



