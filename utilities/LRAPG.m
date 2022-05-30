% Majorized Vector Accelerated Proximal Gradient(MMAPG) Minimization.

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

function Theta_hat = LRAPG(y, X, type, lambda, tol, maxiter, Theta_init)

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
continuation_scaling = 1e-4;
epsilon = 0;
weight = zeros(nr, N);
weight_new = weight;
Theta_old = Theta_init;
Theta_new = Theta_init;
Theta_nold = zeros(nr, nc);
Sampling = zeros(nr, N);
Sampling_new = zeros(nr, N);
G = zeros(nr, nc);
tk_old = 1; tk_new = 1;
tauk = 1e-4 * Lf;
diff = 0;

lambda_run = lambda / continuation_scaling;





for iter = 1 : maxiter
    if (iter > 1)
        Theta_nold = Theta_new + (tk_old - 1)/tk_new * (Theta_new - Theta_old);
        Theta_old = Theta_new;
        tk_old = tk_new;
    end
    
    
    % finding the gradient
    Grad = zeros(nr, nc);
    Sampling = Theta_nold * X;
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
        Grad = (GradMat * X') / (nr);
    end

    taum = eta * tauk;
    while taum < Lf
        % Thresholding
        if verbose == 1
            disp(['Current taum: ', num2str(taum)]);
        end
        G = Theta_nold - Grad/taum;
        
%       [U, S, V, ~] = proxsolver(G, 5, lambda_run/taum);
%       Theta_new = U*S*V';
        Theta_new = sign(G).*max(abs(G)-lambda_run/taum, 0);
        
        Sampling_new = Theta_new * X;
        ResMat_new = y - Sampling_new;
        
        if strcmp(type.name, 'Wilcoxon')
            [~, weight_new] = sort(ResMat_new, 2);
            [~, weight_new] = sort(weight_new, 2);
            weight_new = (weight_new/(N+1) - 0.5)/nr;
            loss_obj = sum(sum(weight_new .* ResMat_new));
            loss_apr = sum(sum(weight .* ResMat)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
                taum/2 * sum(sum((Theta_new - Theta_nold).^2));
        else
            loss_obj = sum(sum(Loss(y - Sampling_new)))/(nr);
            loss_apr = sum(sum(Loss(y - Sampling)))/(nr) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
                taum/2 * sum(sum((Theta_new - Theta_nold).^2));
        end
        
        if verbose == 1
            disp(['loss_obj: ', num2str(loss_obj), ' loss_apr: ', num2str(loss_apr), ' type: ', type.name]);
        end
        if loss_obj < loss_apr
            %          tauk = taum;
            break;
        end
        taum = taum/eta;
    end
    
    
    % Thresholding
    G = Theta_nold - Grad/taum;
%    [U, S, V, rank] = proxsolver(G, 5, lambda_run/taum);
    
    Theta_new = sign(G).*max(abs(G)-lambda_run/taum, 0);
    
    if ~strcmp(type.name, 'Wilcoxon')
        tauk = taum * eta^1;
    else
        tauk = taum * eta^1;
    end
    
    % updating parameters
    tk_new = (1+sqrt(1+4*tk_old^2))/2;
    
    
    diff = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    disp(' ');
    disp(['Iteration: ', num2str(iter), ' obj: ', num2str(loss_obj), ...
        ' apr: ', num2str(loss_apr),...
        ' diff: ', num2str(diff), ...
        ' taum: ', num2str(taum), ' tk: ', num2str(tk_new)]);
    disp(['total_obj: ', num2str(loss_obj+lambda_run*norm(Theta_new)), ' lambda: ', num2str(lambda_run)]);
    disp(' ');
    
    lambda_run = max(lambda_run*0.7, lambda);
    
    if (diff < tol) && (lambda_run == lambda)
        break;
    end
end
Theta_hat = Theta_new;








