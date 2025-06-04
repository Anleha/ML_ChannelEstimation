function [x0, x1, x2, x3]= layer_map(data, length, numTx)
x0=[]; x1=[];
x2=[]; x3=[];
for i=1:length
    if mod(i-1,numTx) == 0
        x0 = [x0 data(i)];
    elseif mod(i-1,numTx) == 1
        x1 = [x1 data(i)];
    elseif mod(i-1,numTx) == 2
        x2 = [x2 data(i)];
    else 
        x3 = [x3 data(i)];
    end
end
    
