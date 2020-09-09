function [X_train, X_test, L_train, L_test] = split(data, labels, n_per, k)

X_train = []; X_test = [];  L_train =[]; L_test=[];
class = max(labels);
for i=1:class
    pos = find(labels==i);
    rng(2020*k+n_per+i);  
    idx = randperm(length(pos));
    T = data(:,pos(idx));
    L = labels(pos);
    X_train = [X_train T(:,1:n_per)];
    X_test  = [X_test  T(:,n_per+1:length(pos))];
    L_train = [L_train L(1:n_per)];
    L_test  = [L_test  L(n_per+1:length(pos))];
end


end

