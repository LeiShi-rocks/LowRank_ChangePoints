% Majorized Vector Accelerated Proximal Gradient(MVAPG) Minimization.

% This function minimize: sum 1/2 loss(yi - <Xi, Theta>) + lambda || Theta ||_*
% Input:
% y: a matrix of y_i observations
% X: a 3-dim array contains all the X_i, where i is the third dimension

% lambda : the penalty factor
% type is a struct containing the following fields:
% - name: what type of loss is chosen
% - eta: the rate of updating curvature
% - Lf : the maximal curvature
% - para: parameters for the loss(only needed by Huber)
% tol and maxiter

% Output:
% Theta.hat

function [Theta_hat, rank, outInfo] = MVAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init)

% Y = A0*X: Y: nr x N, X: nc x N, A0: nr x nc

% nr = 40;
% nc = 40;

% r = 5;
% X = zeros(N, p);
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

% Loss = @(epsilon) epsilon.^2;
% Loss = @(epsilon) abs(epsilon);

verbose = 0;
mildverbose = 0;

if strcmp(type.name, 'L2')
    Loss = @(x) x.^2;
    grad = @(x) 2.*x;
elseif strcmp(type.name, 'L1')
    Loss = @(x) abs(x);
    grad = @(x) sign(x);
elseif strcmp(type.name, 'Huber')
    para = type.para;
    Loss = @(x) x.^2.*(abs(x)<=para) + 2.*para.*abs(x).*(abs(x)>para);
    grad = @(x) 2.*x.*(abs(x)<=para) + sign(x).*2.*para.*(abs(x)>para);
elseif strcmp(type.name, 'Wilcoxon')
    Loss = @(x) abs(x);
    grad = @(x) sign(x);
else
    error('No such loss!');
end

dimX = size(X);
dimY = size(y);
moreInfo = 0;
if isfield(type, 'moreInfo') && type.moreInfo 
    moreInfo = 1;
end

% error check
if dimX(2) ~= dimY(2)
    error('Wrong Input!');
end

% Initialization:
nr = dimY(1);
nc = dimX(1);
N = dimX(2);


eta = type.eta;
Lf = type.Lf;
continuation_scaling = 1e-3;

Theta_old = Theta_init;
Theta_new = Theta_init;

tk_old = 1; tk_new = 1;

lambda = Clambda * sqrt((nr + nc)*N)/nr;
lambda_run = lambda / continuation_scaling;
%lambda_run = lambda;
taumax          = Lf; 
tau  =  taumax;
taumin          = 1e-3 * taumax;

for iter = 1 : maxiter
    % calculate required ingradients at Theta_k
    Theta_nnew = Theta_new + (tk_old - 1)/tk_new .* (Theta_new - Theta_old); 
    
    % finding the gradient
    Sampling = Theta_nnew * X;
    ResMat = y - Sampling; % Matrix of residuals
    
    % weight = weighting(y - Sampling, 'L1');
    if strcmp(type.name, 'Wilcoxon')
        [~, weight] = sort(ResMat, 2); % more precisely this is not weight but pre-rank
        [~, weight] = sort(weight, 2);
        weight = (weight/(N+1) - 0.5)/nr;
    else
        weight = ones(nr, N)/nr;
    end
    
    if strcmp(type.name, 'Wilcoxon')
        Grad = -weight * X';
    else
        GradMat = grad(-ResMat);
        Grad = (GradMat * X') / nr;
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
        break;
    end
    
    Sampling_new = Theta_new * X;
    ResMat_new = y - Sampling_new;
    
    if strcmp(type.name, 'Wilcoxon')
        [~, weight_new] = sort(ResMat_new, 2);
        [~, weight_new] = sort(weight_new, 2);
        weight_new = (weight_new/(N+1) - 0.5)/nr;
        obj = sum(sum(weight_new .* ResMat_new)) + lambda_run .* sum(diag(S));
        obj_line = sum(sum(weight .* ResMat)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
            tau/2 * sum(sum((Theta_new - Theta_nold).^2)) + lambda_run .* sum(diag(S));
    else
        obj = sum(sum(Loss(y - Sampling_new)))/nr + lambda_run .* sum(diag(S));
        obj_line = sum(sum(Loss(y - Sampling)))/nr + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
            tau/2 * sum(sum((Theta_new - Theta_nold).^2)) + lambda_run .* sum(diag(S));
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





