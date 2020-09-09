
clear;clc;

load ./data/PIE_32x32.mat
labels = gnd';
data = double(fea)'/255;

class = max(labels);
n_per = 10;
REPEAT = 10;
ACC = [];

param.alpha = .01;  
param.beta= .1;
param.gamma = .01;
param.dim = class;
opts.mu = 1;
opts.rho = 1.1;
opts.Max_Iter = 30;

for repeat=1:REPEAT 
    
    [X_train, X_test, L_train, L_test] = split(data, labels, n_per, repeat);
    X_train = X_train ./repmat(sqrt(sum(X_train .*X_train )), [size(X_train , 1), 1]);
    X_test  = X_test ./repmat(sqrt(sum(X_test .*X_test)), [size(X_test, 1), 1]);

    [W] = PRDR(X_train, L_train, param, opts);
    fea_train = W*X_train;
    fea_test  = W*X_test;

    tr_n = fea_train./repmat(sqrt(sum(fea_train.*fea_train)), [size(fea_train, 1), 1]);
    tt_n = fea_test./repmat(sqrt(sum(fea_test.*fea_test)), [size(fea_test, 1), 1]);
    [pred, nn_index, accuracy] = KNN(1,tr_n',L_train,tt_n',L_test);
    ACC= [ACC accuracy*100];

    fprintf('%d / %d  acc: %.2f \n', repeat, REPEAT, accuracy*100);
end

fprintf('Mean: %.2f  std: %.2f \n', mean(ACC), std(ACC));

