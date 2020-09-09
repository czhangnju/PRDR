function [W] = PRDR(X, label, param, opts)

% min_{W,V, U} 0.5*||V-WX||_F^2 + alpha/2* ||V'*U-Y'*Y||_F^2 + beta/2*
% Tr(WXLX'W') + gamma/2*||W||_F^2
% s.t. ||V_i||=1, V = U
%
alpha = param.alpha; 
beta = param.beta;
gamma = param.gamma;
dim = param.dim;
n = length(label);
mu = opts.mu;
rho = opts.rho;
mu_max = 1e8;
epsilon = 1e-7;
Max_Iter = opts.Max_Iter;

I=eye(max(label));
Y = I(:,label);
W = rand(dim, size(X,1));
V = Y;
%V = rand(dim, n);
U = V;
Z = zeros(size(V));
YY = Y'*Y;
%YY (YY==0) =-1;
XX = X*X';

L = Construct_L(X',label);
P = X*L*X';
H = X'*inv(XX + beta*P + gamma*eye(size(X,1)));

for iter=1:Max_Iter
    % update V
    VA = alpha*(U*U')+ (1+mu)*eye(size(U,1));
    VB = W*X  + alpha*U*YY + mu*U - Z;
    V = VA\VB;
    V = (V./repmat(sqrt(sum(V.*V)),[size(V, 1) 1]));

    % update U
    UA = alpha*(V*V') + mu* eye(size(V,1));
    UB = alpha*V*YY + mu*V+Z;
    U = UA\UB;
    
    % update W
    W = V*H;
    
     % update Z and theta
    Z = Z+mu*(V-U);
    mu= min(mu_max, rho*mu);
    
    obj(iter) = 0.5*norm(V-W*X, 'fro')^2 + alpha/2*norm(V'*U-YY, 'fro')^2 + beta/2*trace(W*P*W') +gamma/2*norm(W,'fro')^2  ;
    
    if iter>3 && abs(obj(iter) - obj(iter-1)) < epsilon
        break;
    end
    
end



end

