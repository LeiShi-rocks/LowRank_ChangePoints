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

function [Theta_hat, rank] = MMAPG(y, X, type, lambda, tol, maxiter, Theta_init)

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

verbose = 0;

mildverbose = 0;

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
continuation_scaling = 1e-4;
epsilon = 0;
weight = zeros(N, 1);
Theta_old = Theta_init;
Theta_new = Theta_init;
Theta_nold = zeros(nr, nc);
Sampling = zeros(N, 1);
Sampling_new = zeros(N, 1);
G = zeros(nr, nc);
tk_old = 1; tk_new = 1;
tauk = 1e-4 * Lf;
diff = 0;

badTauCount = 0;
badTauFlag = 0;

%lambda_run = lambda / continuation_scaling;
lambda_run = lambda;




for iter = 1 : maxiter
    if (iter > 1)
        Theta_nold = Theta_new + (tk_old - 1)/tk_new * (Theta_new - Theta_old);
        Theta_old = Theta_new;
        tk_old = tk_new;
    end
    
    
    % finding the gradient
    Grad = zeros(nr, nc);
    
        
        
%        ind = 0;
%        for indx = 1 : (N-1)
%            for indy = (indx+1):N
%               disp(['running: ', num2str(ind)]);
%               ind = ind + 1;
%               Sampling(ind) = sum(sum(X(:, :, indx).* Theta_nold))-...
%                   sum(sum(X(:, :, indy).* Theta_nold));
%               Grad = Grad + weight(ind) * (X(:,:,indx)-X(:,:,indy)) * grad(Sampling(ind) - y(ind));
%            end
%        end

        for i = 1 : N
            Sampling(i) = sum(sum(X(:, :, i).* Theta_nold));
%            Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
        end

    
            %weight = weighting(y - Sampling, 'L1');
        if strcmp(type.name, 'Wilcoxon') 
            [~,weight] = sort(y-Sampling); % more precisely this is not weight but pre-rank
            [~,weight] = sort(weight); 
        else 
            weight = ones(N,1); 
        end     
        
%     %weight = weighting(y - Sampling, 'L1');
%     weight = ones(N_run,1);
%     
%     % finding the gradient
%     Grad = zeros(nr, nc);
      if strcmp(type.name, 'Wilcoxon')
         for i = 1 : N  
            Grad = Grad - (weight(i)/(N-1)-0.5*(N+1)/(N-1)) * X(:, :, i);
         end 
      else
         for i = 1 : N  
            Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
         end 
      end
%        ind = 0;
%        for indx = 1 : (N-1)
%            for indy = (indx+1):N
%                ind = ind + 1;
%                Grad = Grad + ...
%                    weight(ind) * X(:, :, ind) * (grad(Sampling(ind)) - y(ind));
%            end
%        end
%     end
    
%     options.tol   = 1e-6; 
%     options.issym = true; 
%     options.disp  = 0; 
%     options.v0    = randn(N,1);
%     tauk = eigs(@(y)AAmap(AATmap(y, X, weight), X, weight),N,1,'LM',options);
%     disp(['Estimating tauk: ', num2str(tauk)]);
   
    taum = eta * tauk;
    while taum < Lf
      % Thresholding
      if verbose == 1
        disp(['Current taum: ', num2str(taum)]);
      end
      G = Theta_nold - Grad/taum;
      
      [U, S, V, ~] = proxsolver(G, 3, lambda_run/taum);
      Theta_new = U*S*V';
      
      for i = 1 : N
          Sampling_new(i) = sum(sum(X(:, :, i).* Theta_new));
      end 
      
      if strcmp(type.name, 'Wilcoxon')
        [~,weight_new] = sort(y-Sampling_new);
        [~,weight_new] = sort(weight_new);
        loss_obj = sum((weight_new/(N-1)-0.5*(N+1)/(N-1)).*(y - Sampling_new));
        loss_apr = sum((weight/(N-1)-0.5*(N+1)/(N-1)).*(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
          taum/2 * sum(sum((Theta_new-Theta_old).^2)); % sum(sum((Theta_new-Theta_old).^2))
      else
        loss_obj = sum(Loss(y - Sampling_new));
        loss_apr = sum(Loss(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
          taum/2 * sum(sum((Theta_new-Theta_old).^2));
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
    [U, S, V, rank] = proxsolver(G, 3, lambda_run/taum);
    
    Theta_new = U*S*V';
    
    if ~strcmp(type.name, 'Wilcoxon')
        tauk = taum * eta^1;
    else
        tauk = taum * eta^1;
    end
    
    % updating parameters
    tk_new = (1+sqrt(1+4*tk_old^2))/2;
    
    
    diff = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    if mildverbose
        disp(' ');
        disp(['Iteration: ', num2str(iter), ' obj: ', num2str(loss_obj), ...
            ' apr: ', num2str(loss_apr),...
            ' diff: ', num2str(diff), ...
            ' taum: ', num2str(taum), ' tk: ', num2str(tk_new)]);
        disp(['total_obj: ', num2str(loss_obj+lambda_run*sum(diag(S))), ' lambda: ', num2str(lambda_run)]);
        disp(' ');
    end
    
%     lambda_run = max(lambda_run*0.5, lambda);
%     
%     if (diff < tol) && (lambda_run < lambda+5e-7)
%         break;
%     end
    
     if diff < tol
        break;
    end
    
    % check whether maximal tau is hit
    
%     if taum >= Lf
%        if lambda_run < lambda+5e-7 && badTauFlag == 0
%            badTauFlag = 1;
%            badTauCount = badTauCount + 1;
%        elseif lambda_run < lambda+5e-7 && badTauFlag == 1
%            badTauCount = badTauCount + 1;
%            if badTauCount > 5
%                disp('Stop in advance due to hitting the picked maximal curvature!');
%                break;
%            end
%        end
%     else 
%         badTauFlag = 0;
%         badTauCount = 0;
%     end
    
end
Theta_hat = Theta_new;









