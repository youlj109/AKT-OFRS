function [w_B]=truncate_online(w,dimen_t,NumFeature)
B=round(NumFeature*dimen_t);
if sum(w~=0)>B,
    [sw,index] = sort(abs(w),'descend');
    sw(B+1:end)=0;
    [sw1,index1]=sort(index);
    w_B=sw(index1).*sign(w);
else
    w_B=w;
end