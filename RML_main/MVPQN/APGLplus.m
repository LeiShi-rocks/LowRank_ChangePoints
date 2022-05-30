% The original APGL function is not general enough, since it's wrapped for
% least square optimization. We wish to work on a general quadratic form.

% solve optimization problem of the form:
% F = f + lambda * ||Theta||_*, where f is a quadratic function.

function [Theta_hat, rank] = APGLplus(f_init, Grad_init, Bk, Theta_init, lambda, opts)

% opts: a struct with field:
%  - lambda  : the penalization factor
%  - tol     : tolerance for stop     
%  - maxiter : maximum steps of iteration
%  - eta     : the decaying factor in curvature line search
%  - continuation_scaling : continuation scaling factor
%  - verbose : display progress or not

%% Boring Initialization
smoothf = @(Theta) f_init + ...
    sum(sum(Grad_init.*(Theta - Theta_init))) + ...
    0.5 .* sum(sum((Theta - Theta_init) .* Bk((Theta - Theta_init))));

dim       =  size(Theta_init);
nr        =  dim(1);
nc        =  dim(2);
Theta_old =  Theta_init;
Theta_new =  Theta_init;
Grad      =  zeros(nr, nc);
% lambda    =  setOpts(opts, 'lambda', 1.0);
tol       =  setOpts(opts, 'tol', 1e-4);
maxiter   =  setOpts(opts, 'maxiter', 100);
eta       =  setOpts(opts, 'eta', 0.8);
verbose   =  setOpts(opts, 'verbose', 1);
continuation_scaling = setOpts(opts, 'continuation_scaling', 1e-3);
lambda_run  =  lambda / continuation_scaling;
tk_old  =  1; tk_new = 1;

if verbose
    disp('Running APGLplus...');
end

%% estimate the Lipchitz constant
nrnc            = nr * nc;
Bk_vec          = @(vTheta) reshape(Bk(reshape(vTheta, [nr, nc])), [nrnc, 1]);
options.tol     = 1e-6;
options.issym   = true;
options.disp    = 0;
options.v0      = randn(nrnc,1);
Lipschitz_const = eigs(Bk_vec, nrnc, 1, 'largestabs', options);
% Lipschitz_const = full(Lipschitz_const);
taumax          = Lipschitz_const; tau  =  taumax;
taumin          = 1e-3 * taumax;

%% Start optimization
for iter = 1 : maxiter
    % calculate required ingradients at Theta_k
    Theta_nnew = Theta_new + (tk_old - 1)/tk_new .* (Theta_new - Theta_old);
    Grad = Grad_init + Bk( Theta_nnew - Theta_init ); % Grad at Theta_k
    
    % calculate Theta_k+1
    Theta_old = Theta_new; % Theta_old = Theta_k
    Theta_nold = Theta_nnew;
    G         = Theta_nnew - Grad/tau;
    
    [U, S, V, rank] = proxsolver(G, 5, lambda_run/tau);

    Theta_new = U*S*V';
    
    % check stop
    diff_norm = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    if (diff_norm < tol) && (lambda_run == lambda)
        break;
    end
    
    % update curvature using line search
    obj       =  smoothf(Theta_new) + lambda_run .* sum(diag(S));
    obj_line  =  smoothf(Theta_nold) + lambda_run .* sum(diag(S)) + ...
        sum(sum( Grad .* (Theta_new - Theta_nold) )) + ...
        (tau/2) .* sum(sum( (Theta_new - Theta_nold).^2 ));
    if obj < 0.9999 * obj_line
        tau = min(taumax,max(eta*tau, taumin)); 
    elseif obj > obj_line
        % restart using Theta_old
        Theta_new  =  Theta_old;
        if (lambda_run == lambda)
            taumin = taumin/eta;
        end
        tau  = min(tau/eta, taumax);
        tk_new = 1;
    end

    % update intermediate parameters
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
if verbose
    disp('verbose');
end