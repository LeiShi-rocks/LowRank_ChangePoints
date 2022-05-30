% Majorized Matrix Accelerated Proximal Gradient(MMAPG) Minimization for sparse matrix completion.


function [Theta_hat, svp, outInfo] = MMAPG_sp(y, II, JJ, opts) % opts: type, lambda, tol, maxiter, Theta_init

%% Initialization:

clear global  %% important 
global Thetak Thetakk Thetakkk Grad 

verbose      =  setOpts(opts, 'verbose', 0);
mildverbose  =  setOpts(opts, 'mildverbose', 0);
nr           =  setOpts(opts, 'nr', 100);
nc           =  setOpts(opts, 'nc', 100);
N            =  setOpts(opts, 'N' , 1000);
lambda       =  setOpts(opts, 'lambda', 1.0);
eta          =  setOpts(opts, 'eta', 0.9);
tol          =  setOpts(opts, 'tol', 1e-4);
Lf           =  setOpts(opts, 'Lf', 1e5);
maxiter      =  setOpts(opts, 'maxiter', 100);
continuation_scaling = setOpts(opts, 'continuation_scaling', 1e-4);
matFormat    =  setOpts(opts, 'matFormat', 'standard');

if strcmp(matFormat, 'factor')
    Theta_init_dft.U = sparse(nr,1);
    Theta_init_dft.S = 0;
    Theta_init_dft.V = sparse(nc,1);
    Theta_init   =  setOpts(opts, 'Theta_init', Theta_init_dft);
else
    Theta_init   =  setOpts(opts, 'Theta_init', zeros(nr, nc));
end

type_dft     =  struct(...
    'name', 'L2', ...
    'para', 1.0);
type         =  setOpts(opts, 'type', type_dft);

taumax       =  Lf;
taumin       =  1e-4 * taumax;
tauk         =  taumin;
taum         =  taumax;

Thetak      =  Theta_init;
Thetakk     =  Theta_init; % may be sparse. Could we use factorization to accelerate this?

Sampling     =  zeros(N, 1);
Sampling_new =  zeros(N, 1);
spGrad       =  zeros(N, 1); % sparse subgradient

tk           =  1;     %  for moment acceleration
tkk          =  1;     %  for moment acceleration

svp          =  0;     %  current positive singular values
sv           =  1;     %  predetermined rank
% svpold       =  svp;
% svold        =  sv;

% G            =  zeros(nr, nc);
diff         =  0;
lambda_run   =  lambda ;%/ continuation_scaling;
sqrt_nr_nc   =  sqrt(nr*nc);
nmin         =  min(nr, nc);
II_JJ       =  sub2ind([nr, nc], II, JJ);
svdtol       =  1.0e-6;

if strcmp(type.name, 'L2')
    Loss = @(x) x.^2;
    grad = @(x) 2.*x;
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


%% start optimization

if (mildverbose)
    fprintf('\n Start running MMAPG...');
    fprintf('\n  nr = %2.0d, nc = %2.0d, N = %2.0d',nr,nc,N);
    fprintf('\n  Ceiling curvature = %3.2e', Lf);
    fprintf('\n  lambda = %3.2e', lambda);
    fprintf('\n-----------------------------------------------');
    fprintf('-------------------------------------------');
    fprintf('\n iter     lambda      taum      svp  ');
    fprintf(' diff_norm     obj       obj_line')
    fprintf('\n-----------------------------------------------');
    fprintf('-------------------------------------------');
end

tic;
for iter = 1 : maxiter
 
    
    %% calculate required ingradients at Theta_k
    beta1  =  1 + (tk - 1)/tkk;
    beta2  =  (tk - 1)/tkk;
    
    if strcmp(matFormat, 'standard')
        Thetakka = beta1 * Thetakk - beta2 * Thetak; % augmented Theta_k
        Thetakka_active = Thetakka(II_JJ);
        Sampling   = Thetakka_active * sqrt_nr_nc;
    else
        Ukk               =   Thetakk.U;
        Vkk               =   Thetakk.V;
        Skk               =   Thetakk.S;
        Ukk_active        =   Ukk(II, :);
        Vkk_active        =   Vkk(JJ, :);
        Uk                =   Thetak.U;
        Vk                =   Thetak.V;
        Sk                =   Thetak.S;
        Uk_active         =   Uk(II, :);
        Vk_active         =   Vk(JJ, :);
        Thetakk_active    =   sum(Ukk_active .* Vkk_active .* diag(Skk)', 2);
        Thetak_active     =   sum(Uk_active .* Vk_active .* diag(Sk)', 2);
        Thetakka_active   =   beta1 * Thetakk_active - beta2 * Thetak_active;
        Sampling          =   Thetakka_active * sqrt_nr_nc;
    end
    
    if strcmp(type.name, 'Wilcoxon')
        [sortRes, sortInd] = sort(y-Sampling); % more precisely this is not weight but pre-rank
    end
    
    spGrad = zeros(N,1);
    
    if strcmp(type.name, 'Wilcoxon')
        spGrad(sortInd) = - (((1:N)'/(N-1))-0.5*(N+1)/(N-1)) * sqrt_nr_nc;
    else
        spGrad = sqrt_nr_nc * grad(Sampling - y);
    end
    
    Grad  =  sparse(II, JJ, spGrad, nr, nc);
    
    
    % objective value at k
    if strcmp(type.name, 'Wilcoxon')
        weight =  ((1:N)'./(N-1))-0.5*(N+1)/(N-1);
        obj    =  sum(weight .* sortRes);
    else
        obj    =  sum(Loss(y - Sampling));
    end
    
    tr_Grad_Thetakka = sum(spGrad.*Thetakka_active);
    
    if strcmp(matFormat, 'factor')
        trkk_square      = sum((diag(Skk)).^2);
        trk_square       = sum((diag(Sk)).^2);
        Uk_Ukk_Skk       = (Uk'*Ukk)*Skk;
        Sk_Vk_Vkk        = Sk*(Vk'*Vkk);
        trk_kk           = sum(sum(Uk_Ukk_Skk.*Sk_Vk_Vkk));
    end
    
    
    %% calculate Theta_k+1
    param.beta2 = (tk - 1)/tkk;
    param.beta1 = 1 + param.beta2;
    param.svdtol = svdtol;
    param.matFormat = matFormat;
    param.nr = nr;
    param.nc = nc;
    param.lambda_run = lambda_run;
    param.tau = taum;
    
    % G              =  Thetakka - Grad/taum;
    [Ukkk, Skkk, Vkkk, svp] =  proxsolver_sp(sv, 3, param);
    % Thetak      =  Thetakk; % Theta_old = Theta_k
    if strcmp(matFormat, 'standard')
        Thetakkk = Ukkk*Skkk*Vkkk'; % Theta_k+1
    else
        Thetakkk.U    =  Ukkk;
        Thetakkk.S    =  Skkk;
        Thetakkk.V    =  Vkkk;
    end
    
    
    %% check stop
    if strcmp(matFormat, 'standard')
        diff_norm = norm(Thetakkk-Thetakk, 'fro')/norm(Thetakk, 'fro');
    else
        trkkk_square  = sum((diag(Skkk)).^2);
        Ukk_Ukkk_Skkk = (Ukk'*Ukkk)*Skkk;
        Skk_Vkk_Vkkk  = Skk*(Vkk'*Vkkk);
        trkk_kkk    = sum(sum(Ukk_Ukkk_Skkk.*Skk_Vkk_Vkkk));
        
        diff_norm = sqrt(trkkk_square + trkk_square - 2*trkk_kkk) / sqrt(trkk_square);
    end
    
    if (diff_norm < tol) && (lambda_run == lambda)
        break;
    end
    
    %% update curvature using line search
    
%     for i = 1 : N
%         Sampling_new(i) = Theta_new_run(II(i), JJ(i)) * sqrt_nr_nc; % might be accelerated with factor matrices
%     end
    
    if strcmp(matFormat, 'standard')
        Thetakkk_active =  Thetakkk(II_JJ);
        Sampling_new    =  Thetakkk_active * sqrt_nr_nc;
    else
        Ukkk_active      =  Ukkk(II, :);
        Vkkk_active      =  Vkkk(JJ, :);
        Thetakkk_active  =  sum(Ukkk_active .* Vkkk_active .* diag(Skkk)', 2);
        Sampling_new     =  Thetakkk_active * sqrt_nr_nc;
    end
    
    if strcmp(type.name, 'Wilcoxon')
        [sortRes, ~] = sort(y - Sampling_new); % more precisely this is not weight but pre-rank
        weight       = ((1:N)'./(N-1))-0.5*(N+1)/(N-1);
        obj_new      = sum(weight .* sortRes);
    else
        obj_new      = sum(Loss(y - Sampling_new));
    end
    
    tr_Grad_Thetakkk = sum(spGrad .* Thetakkk_active);
    
    if strcmp(matFormat, 'standard')
        obj_line  =  obj + ...
        tr_Grad_Thetakkk - tr_Grad_Thetakka + ...
        (taum/2) .* sum(sum( (Thetakkk - Thetakka).^2 ));
    else
        Uk_Ukkk_Skkk = (Uk'*Ukkk)*Skkk;
        Sk_Vk_Vkkk   = Sk*(Vk'*Vkkk);
        trk_kkk      = sum(sum(Uk_Ukkk_Skkk.*Sk_Vk_Vkkk));
        
        normkkk_kka_square = trkkk_square + beta1^2*trkk_square + beta2^2*trk_square...
            - 2*beta1*trkk_kkk + 2*beta2*trk_kkk - 2* beta1 * beta2 * trk_kk;
        
        obj_line  =  obj + ...
        tr_Grad_Thetakkk - tr_Grad_Thetakka + ...
        (taum/2) .* normkkk_kka_square;
    end
    
    
    if obj_new <  obj_line
        % disp('good');
        taum = min(taumax, max(eta*taum, taumin));
        % update intermediate parameters
        tk_med  =  tkk;
        tk      =  tkk;
        tkk     =  (1+sqrt(1+4*tk_med^2))/2; % tk
        Thetak  =  Thetakk;
        Thetakk =  Thetakkk;
    elseif obj_new > obj_line
        % restart using Theta_old
        % disp('bad');
        Thetak  =  Thetakk;
        if (lambda_run == lambda)
            taumin = taumin/eta;
        end
        taum  = min(taum/eta, taumax);
        tk = 1; tkk = 1;
    end
    
    lambda_run = max(lambda_run * 0.7, lambda); % lambda_run
    
    %% update predetermined rank
    if (svp == sv) && (abs(lambda_run/lambda) < 50) && (obj_new < 10*obj_line)
        sv = min(svp+5, nmin);
    else
        sv = min(svp+1, nmin);
    end
    
    % display progress if mildverbose == 1
%     if mildverbose
%         disp(['| Iter: ', num2str(iter), ...
%             '| obj: ', num2str(obj), ...
%             '| obj_line: ', num2str(obj_line), ...
%             '| diff: ', num2str(diff_norm), ...
%             '| lambda_run: ', num2str(lambda_run), ...
%             '| taum: ', num2str(taum), ...
%             '| svp: ', num2str(svp), ...
%             '| sv: ', num2str(sv)
%             ]);
%     end
    
    if (mildverbose)
        fprintf('\n %3.0d    %3.2e  %3.2e  |%3.0d %3.0d| ',iter, lambda_run, taum, svp, sv);
        fprintf('%3.2e| %5.4e %5.4e|', diff_norm, obj_new, obj_line);
    end
    
end

Theta_hat = Thetakkk;
elapsedTime = toc;
outInfo.elapsedTime = elapsedTime;
outInfo.numIter     = iter;

if (mildverbose)
    fprintf(1,'\n Finished the main algorithm!\n')
    fprintf(1,' Objective function        = %6.5e\n', obj_new);
    fprintf(1,' Number of iterations      = %2.0d\n', iter);
    fprintf(1,' Number of singular values = %2.0d\n', svp);
    fprintf(1,' CPU time                  = %3.2e\n', elapsedTime);
    fprintf(1,' norm(X-Xold,''fro'')/norm(X,''fro'') = %3.2e\n', diff_norm);
    fprintf(1,'\n');
end

%% Successful version!

% %lambda_run = lambda / continuation_scaling;
% lambda_run = lambda;
% Theta_nold = Theta_init;
% verbose = 1; mildverbose = 1;
% % X = zeros(nr, nc, N);
% % for i = 1:N
% %     X(II(i),JJ(i),i) = sqrt(nr*nc);
% % end
% 
% for iter = 1 : maxiter
%     if (iter > 1)
%         Theta_nold = Theta_new + (tk_old - 1)/tk_new * (Theta_new - Theta_old);
%         Theta_old = Theta_new;
%         tk_old = tk_new;
%     end
% 
%     % finding the gradient
%     % Grad = zeros(nr, nc);
% 
%     for i = 1 : N
% 
%         Sampling(i) = Theta_nold(II(i), JJ(i)) * sqrt_nr_nc;
%         %            Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
%     end
% 
%     if strcmp(type.name, 'Wilcoxon')
%         [~,weight] = sort(y-Sampling); % more precisely this is not weight but pre-rank
%         [~,weight] = sort(weight);
%     else
%         weight = ones(N,1);
%     end
% 
%     if strcmp(type.name, 'Wilcoxon')
%         for i = 1 : N
%             % Grad = Grad - (weight(i)/(N-1)-0.5*(N+1)/(N-1)) * X(:, :, i);
%             spGrad(i) = - (weight(i)/(N-1)-0.5*(N+1)/(N-1)) * sqrt_nr_nc;
%         end
%     else
%         for i = 1 : N
%             % Grad = Grad + weight(i) * X(:, :, i) * grad(Sampling(i) - y(i));
%             spGrad(i) = sqrt_nr_nc * grad(Sampling(i) - y(i));
%         end
%     end
% 
%     Grad = sparse(II, JJ, spGrad, nr, nc);
% 
%     taum = eta * tauk;
% 
%     %% update predetermined rank
%     if (svp == sv) && (abs(lambda_run/lambda) < 50)
%         sv = min(svp+5, nmin);
%     else
%         sv = min(svp+1, nmin);
%     end
% 
%     while taum < Lf
%         % Thresholding
%         if verbose == 1
%             disp(['Current taum: ', num2str(taum)]);
%         end
%         G = Theta_nold - Grad/taum;
% 
%         [U, S, V, ~] = proxsolver_sp(G, sv, 3, lambda_run/taum, svdtol);
%         % [U, S, V, ~] = proxsolver(G, 3, lambda_run/taum);
%         Theta_new = U*S*V';
% 
%         for i = 1 : N
%             Sampling_new(i) = Theta_new(II(i), JJ(i)) * sqrt_nr_nc;
%         end
% 
%         if strcmp(type.name, 'Wilcoxon')
%             [~,weight_new] = sort(y - Sampling_new);
%             [~,weight_new] = sort(weight_new);
%             loss_obj = sum((weight_new/(N-1)-0.5*(N+1)/(N-1)).*(y - Sampling_new));
%             loss_apr = sum((weight/(N-1)-0.5*(N+1)/(N-1)).*(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
%                 taum/2 * sum(sum((Theta_new-Theta_nold).^2)); % sum(sum((Theta_new-Theta_old).^2))
%         else
%             loss_obj = sum(Loss(y - Sampling_new));
%             loss_apr = sum(Loss(y-Sampling)) + sum(sum(Grad .* (Theta_new - Theta_nold))) + ...
%                 taum/2 * sum(sum((Theta_new-Theta_old).^2));
%         end
% 
%         if verbose == 1
%             disp(['loss_obj: ', num2str(loss_obj), ' loss_apr: ', num2str(loss_apr), ' type: ', type.name]);
%         end
%         if loss_obj < loss_apr
%             %          tauk = taum;
%             break;
%         end
%         taum = taum/eta;
%     end
% 
% 
% 
%     % Thresholding
%     G = Theta_nold - Grad/taum;
%     [U, S, V, svp] = proxsolver_sp(G, sv, 3, lambda_run/taum, svdtol);
%     % [U, S, V, svp] = proxsolver(G, 3, lambda_run/taum);
% 
%     Theta_new = U*S*V';
% 
%     if ~strcmp(type.name, 'Wilcoxon')
%         tauk = taum * eta^1;
%     else
%         tauk = taum * eta^1;
%     end
% 
%     % updating parameters
%     tk_new = (1+sqrt(1+4*tk_old^2))/2;
% 
%     diff = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
% 
%     if mildverbose
%         disp(['| Iter: ', num2str(iter), ...
%             '| obj: ', num2str(loss_obj), ...
%             '| obj_line: ', num2str(loss_apr), ...
%             '| diff: ', num2str(diff), ...
%             '| lambda_run: ', num2str(lambda_run), ...
%             '| taum: ', num2str(taum), ...
%             '| svp: ', num2str(svp), ...
%             '| sv: ', num2str(sv)
%             ]);
%     end
% 
% 
%     if diff < tol
%         break;
%     end
% 
% end
% 
% Theta_hat = Theta_new;



























