% Majorized Vector Accelerated Proximal Quasi-Newton Method.

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

% Revision: 0.9.0  || Date: 12/27/2021

function [Theta_hat, rank, outInfo] = MVPQN(y, X, opts)
%% Boring Initialization

dimX = size(X);
dimY = size(y);
outInfo = struct();

% sanity check
if dimX(2) ~= dimY(2)
    error('Wrong Input!');
end


% Initialization:

nr          = dimY(1);
nc          = dimX(1);
N           = dimX(2);
dfttype     = struct(...
    'name', 'L2', ...
    'para', '1.0');
dftHessian  = struct(...
    'name', 'const_diag',...
    'Lf'  , 2e4, ...
    'para', 1);
dftt_search = struct(...
    'name'   , 'trivial',...
    'para' , 1);

verbose     = setOpts(opts, 'verbose', 1);
mildverbose = setOpts(opts, 'mildverbose', 1);

type        = setOpts(opts, 'type', dfttype);
% a struct with a field name:
% - if name is 'L2', use L2 loss. No additional parameter required. But for
% uniformality, set para = 1.0.
% - if name is 'L1', use L1 loss. No additional parameter required. But for
% uniformality, set para = 1.0.
% - if name is 'Huber', use Huber loss. Require another field para to
% specify the turning point.
% - if name is 'Wilcoxon', use Wilcoxon loss. No additional para required. But for
% uniformality, set para = 1.0.

Hessian     = setOpts(opts, 'Hessian', dftHessian);
% a struct with a field name:
% - if name is 'const_diag', use first-order method; an additional field is
% Lf, specifying the ceiling curvature
% - if name is 'LSR1', use limited-memory rank-1 update for Hessian; an additional field
% is para, specifying how many previous steps to store.
% - if name is 'LBFGS', use limited-memory BFGS update; an additional field
% is para, specifying how many previous steps to store.

t_search    = setOpts(opts, 't_search', dftt_search);
% a struct with a field name:
% - if name is 'trivial', use a constant fraction of the increment. Require an
% additional field para to spercify the fraction. Default is 1.
% - if name is 'line_search', use line search as Lee2014.  Create the
% field para(0~0.5 as Lee2014 suggested) to spercify fraction of decreasing anticipated

approx_flag    =  0;   % used to mark the start of Hessian approximation
lambda         =  setOpts(opts, 'lambda', 1.0);
tol            =  setOpts(opts, 'tol', 1e-4);
maxiter        =  setOpts(opts, 'maxiter', 200);
Theta_init     =  setOpts(opts, 'Theta_init', zeros(nr, nc));
eta            =  setOpts(opts, 'eta', 0.8);
continuation_scaling = setOpts(opts, 'continuation_scaling', 1e-3);

Lf             =  Hessian.Lf;
weight         =  zeros(nr, N);
weight_new     =  weight;
Theta_old      =  Theta_init;
Theta_new      =  Theta_init;
Sampling       =  zeros(nr, N);
Sampling_new   =  zeros(nr, N);
Grad_old       =  zeros(nr, nc);
diff           =  0;
lambda_run     =  lambda / continuation_scaling;



% Construct the loss function
switch type.name
    
    case 'L2'
        Loss = @(x) x.^2;
        grad = @(x) 2.*x;
        
    case 'L1'
        Loss = @(x) abs(x);
        grad = @(x) sign(x);
        
    case 'Huber'
        para = type.para;
        Loss = @(x) x.^2.*(abs(x)<=para) + 2.*para.*abs(x).*(abs(x)>para);
        grad = @(x) 2.*x.*(abs(x)<=para) + sign(x).*2.*para.*(abs(x)>para);
        
    case 'Wilcoxon'
        Loss = @(x) abs(x);
        grad = @(x) sign(x);
        
    otherwise
        error('No such loss defined. Make changes if you wish.');
        
end

%% Begin optimization

if strcmp(Hessian.name, 'const_diag')
    if verbose
        disp('Constant diagonal Hessian approximation!');
    end
    newOpts = struct(...
        'name', type.name, ...
        'eta' , eta, ...
        'Lf'  , Lf,  ...
        'para', type.para, ...
        'moreInfo', 1);
    [Theta_hat, outputs] = MVAPG(y, X, newOpts, lambda, tol, maxiter, Theta_init);
    rank = outputs(1);
    outInfo.TotalIteration = outputs(2);
else
    ls    =  Hessian.para;  % limited storage steps
    hDiag =  Hessian.Lf;
    SIn   =  zeros(nr, nc, 0); % for increment of Theta
    YIn   =  zeros(nr, nc, 0); % for increment of gradient
    for iter = 1 : maxiter
        
        %% finding the gradient
        Sampling = Theta_new * X;
        ResMat   = y - Sampling; % Matrix of residuals
        
        if strcmp(type.name, 'Wilcoxon')
            [~, weight] = sort(ResMat, 2); % more precisely this is not weight but pre-rank
            [~, weight] = sort(weight, 2);
            weight      = (weight/(N+1) - 0.5)/nr;
        else
            weight      = ones(nr, N)/nr;
        end
        
        if strcmp(type.name, 'Wilcoxon')
            Grad    = -weight * X';
        else
            GradMat = grad(-ResMat);
            Grad    = (GradMat * X') / nr; % since weight is not used here, so rescaling is needed.
        end
        
        %% find the Hessian approximation
        switch Hessian.name
            case 'LBFGS'
                if verbose
                    disp('Limited memory BFGS Hessian approximation!');
                end
                if iter > 1 && lambda_run == lambda
                    approx_flag = 1;
                    sIn =  Theta_new - Theta_old;
                    yIn = Grad - Grad_old;
                    if sum(sum( yIn .* sIn )) > 1e-9
                        % disp('Updating hDiag...');
                        if size( SIn, 3 ) >= ls
                            SIn   = cat(3, SIn(:,:,2:ls), sIn);
                            YIn   = cat(3, YIn(:,:,2:ls), yIn);
                            hDiag = sum(sum( yIn .* yIn )) / sum(sum( yIn .* sIn ));
                        else
                            SIn   = cat(3, SIn, sIn);
                            YIn   = cat(3, YIn, yIn);
                            hDiag = sum(sum( yIn .* yIn )) / sum(sum( yIn .* sIn ));
                        end
                        % else
                        %    Bk    = @(Theta) hDiag .* Theta;
                    end
                    Bk  = MVPQN_Hessian(SIn, YIn, hDiag, 'LBFGS');
                else
                    SIn   = zeros(nr, nc, 0 );
                    YIn   = zeros(nr, nc, 0 );
                    hDiag = max(hDiag * eta, 1e-2*Lf);
                    Bk    = @(Theta) hDiag .* Theta;
                end % Bk gives Hessian(k)
            case 'LSR1'
                if verbose
                    disp('Limited memory SR1 Hessian approximation!');
                end
                if iter > 1 && lambda_run == lambda
                    approx_flag = 1;
                    sIn =  Theta_new - Theta_old;
                    yIn = Grad - Grad_old; % BB step: Barzilai-Borwein step length
                    vIn = yIn - 0.4 * sum(sum( sIn .* yIn )) ./ sum(sum( sIn .* sIn )) .* sIn;
                    % gaurantee positiveness
                    if verbose
                        disp(['output: ', num2str(sum(sum( vIn .* sIn ))), ...
                        ' ', num2str(1e-9 * norm(vIn, 'fro') * norm(sIn, 'fro'))]);
                    end
                    if sum(sum( vIn .* sIn )) > 1e-9 * norm(vIn, 'fro') * norm(sIn, 'fro')
                        if verbose
                            disp('Updating hDiag...');
                        end
                        if ls > 1
                            error('Only 1 step storage SR1 is implemented!');
                        else
                            SIn   = sIn;
                            YIn   = yIn;
                            hDiag = sum(sum( sIn .* yIn )) / sum(sum( sIn .* sIn ));
                            % disp(num2str(sign(sum(sum( yIn .* sIn )))));
                            % % should be positive for convex problems.
                        end
                        Bk  = MVPQN_Hessian(SIn, YIn, hDiag, 'LSR1');
                    % else
                    %   hDiag = max(hDiag * eta, 1e-2*Lf);
                    %   Bk    = @(Theta) hDiag .* Theta;
                        %    Bk    = @(Theta) hDiag .* Theta;
                    end
                else
                    SIn   = zeros(nr, nc, 0 );
                    YIn   = zeros(nr, nc, 0 );
                    hDiag = max(hDiag * eta, 1e-2*Lf);
                    Bk    = @(Theta) hDiag .* Theta;
                end % Bk gives Hessian(k)
            otherwise
                disp('No such method defined!');
        end
        
        
        % Update other info at Theta(k)
        Theta_old    =  Theta_new; % Theta(k)
        Grad_old     =  Grad;      % Grad(k)
        Sampling_new =  Theta_new * X;
        ResMat_new   =  y - Sampling_new;
        if strcmp(type.name, 'Wilcoxon')
            [~, weight_new] = sort(ResMat_new, 2);
            [~, weight_new] = sort(weight_new, 2);
            weight_new      = (weight_new/(N+1) - 0.5)/nr;
            loss_obj        = sum(sum(weight_new .* ResMat_new));
        else
            loss_obj        = sum(sum(Loss(y - Sampling_new)))/nr;
        end % loss_obj at Theta(k)
        
        %% solve the subproblem
        if approx_flag
            subOpts    = struct(...
            'tol'   , tol, ...
            'maxiter', maxiter, ...
            'eta'   , eta, ...
            'verbose', 0 ...
            );
            [Theta_new, rank]  =  APGLplus(loss_obj, Grad_old, Bk, Theta_old, lambda_run, subOpts);
        else
            G         = Theta_old - Grad_old/hDiag;
            [U, S, V, rank] = proxsolver(G, 5, lambda_run/hDiag);
            Theta_new = U * S * V';
        end
        
        
        %% backtracking line search to determine the sufficient stepsize
        if strcmp(t_search.name, 'line_search')
            if verbose
                disp('Doing line search...');
            end
            alpha = t_search.para;
            diff_Theta   = Theta_new - Theta_old;
            diff_penalty = lambda_run * (sum(svd(Theta_new)) - sum(svd(Theta_old)));
            desc_bar     = sum(sum(Grad.*diff_Theta)) + diff_penalty;
            Theta_test   = Theta_new;
            if iter > 1
                t = 1;
                while t > tol
                    Theta_test    = Theta_old + t * diff_Theta;
                    Sampling_test = Theta_test * X;
                    ResMat_test   = y - Sampling_test;
                    if strcmp(type.name, 'Wilcoxon')
                        [~, weight_test] = sort(ResMat_test, 2);
                        [~, weight_test] = sort(weight_test, 2);
                        weight_test      = (weight_test/(N+1) - 0.5)/nr;
                        loss_obj_test    = sum(sum(weight_test .* ResMat_test));
                    else
                        loss_obj_test    = sum(sum(Loss(y - Sampling_test)))/nr;
                    end
                    desc_obj  =   loss_obj_test - loss_obj + lambda_run * (sum(svd(Theta_test)) - sum(svd(Theta_old)));
                    if desc_obj < alpha * t * desc_bar
                        break;
                    end
                    t = t * 0.5;
                end
                % else
                %    [ x, f_x, grad_g_x, step, backtrack_flag, backtrack_iters ] = ...
                %        pnopt_curvtrack( x, p, max( min( 1, 1 / norm( grad_g_x ) ), xtol ), f_x, ...
                %        grad_g_x'*p, smoothF, nonsmoothF, desc_param, xtol, max_fun_evals - fun_evals );
            end
            Theta_new  =  Theta_test;
        end
        
        
        %% collect data for output
        diff      = norm(Theta_new - Theta_old, 'fro') / norm(Theta_old, 'fro');
        
        if mildverbose
            disp(' ');
            disp(['Iteration: ', num2str(iter), ...
                ' obj: ', num2str(loss_obj), ...
                ' diff: ', num2str(diff), ...
                ' hDiag: ', num2str(hDiag) ...
                ]);
            disp(['total_obj: ', num2str(loss_obj+lambda_run*sum(svd(Theta_new))), ...
                ' lambda: ', num2str(lambda_run)]);
            disp(' ');
        end
        
        %% updating parameters(if needed)
        
        lambda_run = max(lambda_run * 0.7, lambda);
        
        %% check stopping criteria
        
        diff_norm = sqrt(sum(sum( (Theta_new-Theta_old).^2 )) / sum(sum( Theta_old.^2 )));
        if (diff_norm < tol) && (lambda_run == lambda)
            break;
        end
        
        %         if norm( x - x_old, 'inf' ) / max( 1, norm( x_old, 'inf' ) ) <= xtol
        %             flag    = FLAG_XTOL;
        %             message = MESSAGE_XTOL;
        %             loop    = 0;
        %         elseif abs( f_old - f_x ) / max( 1, abs( f_old ) ) <= ftol
        %             flag    = FLAG_FTOL;
        %             message = MESSAGE_FTOL;
        %             loop    = 0;
        %         elseif iter >= max_iter
        %             flag    = FLAG_MAXITER;
        %             message = MESSAGE_MAXITER;
        %             loop    = 0;
        %         end
        %
        %         if (diff < tol) && (lambda_run == lambda)
        %             break;
        %         end
        
        %% display progress if verbose == 1
    end
    
    %% clean up and exit
    outInfo.TotalIteration = iter;
    Theta_hat = Theta_new;
    
end






