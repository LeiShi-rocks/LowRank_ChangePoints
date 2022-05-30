% Majorized Matrix Accelerated Proximal Gradient(MMAPG) Minimization.

% This function minimize: sum 1/2 loss(yi - <Xi, Theta>) + lambda || Theta ||_*
% Input:
% y: a vector of y_i observations
% X: a 3-dim array contains all the X_i, where i is the third dimension
% Loss: a function handle specifying the loss
% lambda : the penalty factor
% eta: accelarate APG by using a smaller curvature constant
% Lf: Lipchitz constant
% tol and maxiter
% Output:
% Theta.hat

function [Theta_hat, rank, outInfo] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init)

% nr = 40;
% nc = 40; 
% N = 2000;
% r = 5;
% X = zeros(nr, nc, N);
% y = zeros(N, 1);
% noise = 0.1;
% lambda = 50;
% eta = 0.5;
% Lf = 6;
% tol = 1e-3; maxiter = 20;
% Theta_init = zeros(nr, nc);
% 
% 
% Theta_star = normrnd(0,1,[nr, r]) * normrnd(0,1,[r, nc]);

% for i = 1 : N
%    % disp(['Currently generating sample: ', num2str(i)]);
%     row_index = ceil(nr * rand(1));
%     col_index = ceil(nc * rand(1));
%     X(row_index, col_index, i) = 1;
%     y(i) = Theta_star(row_index, col_index) + noise * normrnd(0, 1);
% end

%Loss = @(epsilon) epsilon.^2;
%Loss = @(epsilon) abs(epsilon);

verbose = 1;

mildverbose = 0;
moreInfo = 0;
if isfield(type, 'moreInfo') && type.moreInfo 
    moreInfo = 1;
end

if strcmp(type.name, 'L2')
    Loss = @(x) x.^2;
    grad = @(x) 2*x;
elseif strcmp(type.name, 'L1')
    Loss = @(x) abs(x);
    grad = @(x) sign(x);
elseif strcmp(type.name, 'Huber')
    para = type.para;
    Loss = @(x) x.^2.*(abs(x)<=para) + (2.*para.*abs(x)-para.^2).*(abs(x)>para);
    grad = @(x) 2*x*(abs(x)<=para) + sign(x)*2*para*(abs(x)>para);
elseif strcmp(type.name, 'Wilcoxon')
    Loss = @(x) abs(x);
    grad = @(x) sign(x);
else
    error('No such loss!');
end

dim = size(X);

% Initialization:
nr = dim(1);
nc = dim(2);
N = dim(3);


eta = type.eta;
Lf = type.Lf;
continuation_scaling = 1e-3;
Theta_old = Theta_init;
Theta_new = Theta_init;
Sampling = zeros(N, 1);
Sampling_new = zeros(N, 1);
tk_old = 1; tk_new = 1;

lambda = Clambda * sqrt((nr + nc)*N);
lambda_run = lambda / continuation_scaling;
% lambda_run = lambda;
taumax          = Lf; tau  =  taumax;
taumin          = 1e-3 * taumax;



for iter = 1 : maxiter
    %% calculate required ingradients at Theta_k
    Theta_nnew = Theta_new + (tk_old - 1)/tk_new .* (Theta_new - Theta_old);
    
    % finding the gradient
    Grad = zeros(nr, nc);
    for i = 1 : N
         Sampling(i) = sum(sum(X(:, :, i) .* Theta_nnew));
%        Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
    end
    
    if strcmp(type.name, 'Wilcoxon') 
         [~, weight] = sort(y - Sampling); % more precisely this is not weight but pre-rank
         [~, weight] = sort(weight); 
    else 
         weight = ones(N,1); 
    end   
    
    if strcmp(type.name, 'Wilcoxon')
       for i = 1 : N  
          Grad = Grad - (weight(i)/(N-1) - 0.5*(N+1)/(N-1)) * X(:, :, i);
       end 
    else
       for i = 1 : N  
          Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
       end 
    end
    
    %% calculate Theta_k+1
    Theta_old = Theta_new; % Theta_old = Theta_k
    Theta_nold = Theta_nnew;
    G         = Theta_nnew - Grad/tau;
    
    [U, S, V, rank] = proxsolver(G, 5, lambda_run/tau);

    Theta_new = U*S*V';
    
    %% check stop
    diff_norm = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    if (diff_norm < tol) && (lambda_run == lambda)
        outInfo.stop_flag = "converge";
        break;
    end
    
    
    for i = 1 : N
       Sampling_new(i) = sum(sum(X(:, :, i).* Theta_new));
    end 
      
    if strcmp(type.name, 'Wilcoxon')
       [~,weight_new] = sort(y-Sampling_new);
       [~,weight_new] = sort(weight_new);
       obj      = sum((weight_new/(N-1)-0.5*(N+1)/(N-1)).*(y - Sampling_new)) + lambda_run .* sum(diag(S));
       obj_line = sum((weight/(N-1)-0.5*(N+1)/(N-1)).*(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
          (tau/2) * sum(sum((Theta_new-Theta_nold).^2)) + lambda_run .* sum(diag(S)); % sum(sum((Theta_new-Theta_old).^2))
    else
       obj      = sum(Loss(y - Sampling_new)) + lambda_run .* sum(diag(S));
       obj_line = sum(Loss(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
          (tau/2) * sum(sum((Theta_new-Theta_nold).^2)) + lambda_run .* sum(diag(S));
    end
    
    
    %% update curvature using line search
    if obj < 0.9999 * obj_line
        tau = min(taumax, max(eta*tau, taumin)); 
    elseif obj > obj_line
        % restart using Theta_old
        Theta_new  =  Theta_old;
        if (lambda_run == lambda)
            taumin = taumin/eta;
        end
        tau  = min(tau/eta, taumax);
        tk_new = 1;
    end
    
    

    %% update intermediate parameters
    tk_med = tk_new;
    tk_old = tk_new;
    tk_new = (1+sqrt(1+4*tk_med^2))/2; % tk
    
    lambda_run = max(lambda_run * 0.7, lambda); % lambda_run
    
    % display progress if verbose == 1
    if verbose
        disp(['| Iter: ', num2str(iter), ...
              '| obj: ', num2str(obj), ...
              '| obj_line: ', num2str(obj_line), ...
              '| diff_norm: ', num2str(diff_norm), ...
              '| lambda_run: ', num2str(lambda_run), ...
              ]);
    end
end

Theta_hat = Theta_new;

outInfo.iterations = iter;
outInfo.obj  = obj;








