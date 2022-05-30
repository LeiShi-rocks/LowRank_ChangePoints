%% ============================================ 
% No change point, deterministic thresholds, CS
%  ============================================
clear;clc;
nr = 40; 
nc = 40;
N = 2e3;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'CS';
cp_opts = struct(...
    'num_seg', 1,...
    'pos_seg', 0);
design = [];

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);

Theta_star = outInfo.Theta_star;
threshold_var = outInfo.threshold_var;


%% MMAPG pure
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e5);
Clambda = 0.5;
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(nr, nc);

[Theta_hat, rank] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);

disp([sum(sum((Theta_hat - Theta_star).^2)), rank]);



%% MMAPG Assume single change point
X_new = zeros(2*nr, nc, N);
for i = 1:N
    X_new(:,:,i) = [X(:,:,i); X(:,:,i) .* (threshold_var(i) > 0.5)];
end
Clambda = 0.3;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e4);
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(2*nr, nc);

[Theta_hat, ~] = MMAPG_MCP(y, X_new, type, Clambda, tol, maxiter, Theta_init);

disp(sum(sum((Theta_hat(1:nr,:) - Theta_star(:,:)).^2)));
disp(sum(sum((Theta_hat((nr + 1):(2*nr),:)).^2)));


%% MatLassoSCP
APG_opts = struct(...
    'type', type,...
    'Clambda', 0.3,...
    'tol', 1e-4,...
    'maxiter', 1.5e2,...
    'Theta_init', zeros(2*nr, nc));
[Theta_Delta_hat, tau_hat, obj_path, Delta_path] = MatLassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
figure;
plot(obj_path);
figure;
plot(Delta_path);



%% ===============================================================
% Date generation : No change point, deterministic thresholds, MR
% ================================================================
clear;clc;
nr = 50; 
nc = 50;
N = 2000;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.5);
problem = 'MR';
cp_opts = struct(...
    'num_seg', 1,...
    'pos_seg', 0);
design = struct(...
    'type', 'AR',...
    'para', 0);

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
Theta_star = outInfo.Theta_star;

% MVAPG
C = 1.5;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e3);
Clambda = 1.5;
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(nr, nc);

%[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
[Theta_hat, rank] = MVAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);

disp([sum(sum((Theta_hat - Theta_star).^2)), rank]);

% APGL


%% ============================================
% 3 change points, deterministic thresholds, CS
% =============================================
clear;clc;
nr = 50; 
nc = 50;
N = 2e3;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'CS';
cp_opts = struct(...
    'num_seg', 4,...
    'pos_seg', [0, 0.25, 0.5, 0.75]);
design = [];

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
Theta_star = outInfo.Theta_star;

% MMAPG

type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e4);
Clambda = 1.5;
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(nr, nc);

[Theta_hat, rank] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);

disp([sum(sum((Theta_hat - Theta_star(:,:,1)).^2)), rank]);
disp([sum(sum((Theta_hat - Theta_star(:,:,2)).^2)), rank]);
disp([sum(sum((Theta_hat - Theta_star(:,:,3)).^2)), rank]);
disp([sum(sum((Theta_hat - Theta_star(:,:,4)).^2)), rank]);

disp([sum(sum((Theta_hat - mean(Theta_star, 3)).^2)), rank]);



%% 1 change point, deterministic thresholds, CS
clear;clc;
nr = 40; 
nc = 40;
N = 2e3;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'CS';
cp_opts = struct(...
    'num_seg', 2,...
    'pos_seg', [0, 0.5]);
design = [];

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
Theta_star = outInfo.Theta_star;
threshold_var = outInfo.threshold_var;

%% MMAPG
X_new = zeros(2*nr, nc, N);
for i = 1:N
    X_new(:,:,i) = [X(:,:,i); X(:,:,i) .* (threshold_var(i) > 0.5)];
end
C = 0.3;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e4);
Clambda = 0.3;
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(2*nr, nc);

[Theta_hat, rank] = MMAPG_MCP(y, X_new, type, Clambda, tol, maxiter, Theta_init);

disp(sum(sum((Theta_hat(1:nr,:) - Theta_star(:,:,1)).^2)));
disp(sum(sum((Theta_hat(1:nr,:) + Theta_hat((nr + 1):(2*nr),:) - Theta_star(:,:,2)).^2)));

%% MatLassoSCP
APG_opts = struct(...
    'type', type,...
    'Clambda', 0.3,...
    'tol', 1e-4,...
    'maxiter', 2e2,...
    'Theta_init', zeros(2*nr, nc));
[Theta_Delta_hat, tau_hat, obj_path, Delta_path] = MatLassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
figure;
plot(obj_path);
figure;
plot(Delta_path);

%% =============================================
% 2 change points, deterministic thresholds, MR
% ==============================================
clear;clc;
nr = 40; 
nc = 40;
N = 2000;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'MR';
cp_opts = struct(...
    'num_seg', 3,...
    'pos_seg', [0, 0.33, 0.66]);
design = struct(...
    'type', 'AR',...
    'para', 0);

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
Theta_star = outInfo.Theta_star;
threshold_var = outInfo.threshold_var;

%% MatLassoMCP
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e3);
Clambda_base  = 0.15;
window_length = 0.25;
num_windows   = 10;
cutoff        = 0.2;

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 1e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 25,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 1e2);
SCP_args_2 = struct(...
    "kappa", 0.1,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.15,...
    "tol", 1e-4,...
    "maxiter", 1e2);
    
MCP_opts   = struct();


[post_Theta_hat, post_tau_hat, post_rank, MCP_outInfo] = ...
    MatLassoMCP(y, X, threshold_var, Clambda_base,...
    window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);







%% ============================================
% 1 change point, deterministic thresholds, MR
%  ============================================
clear;clc;
nr = 50; 
nc = 50;
N = 2e3;
r = 5;
noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'MR';
cp_opts = struct(...
    'num_seg', 2,...
    'pos_seg', [0, 0.5]);
design = struct(...
    'type', 'AR',...
    'para', 0);

[y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
Theta_star = outInfo.Theta_star;
threshold_var = outInfo.threshold_var;


%% MVAPG - assume non change point
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e3);
Clambda = 0.15;
tol = 1e-4;
maxiter = 1e2;
Theta_init = zeros(nr, nc);

%[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
[Theta_hat, ~] = MVAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);

disp(sum(sum((Theta_hat(:,:) - Theta_star(:,:,1)).^2)));
disp(sum(sum((Theta_hat(:,:) - Theta_star(:,:,2)).^2)));

%% MVAPG
tau = 0.5;
for i = 1:N
    X_new(:,i) = [X(:,i); X(:,i) .* (threshold_var(i) > tau)];
end

type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e3);
Clambda = 0.15;
tol = 1e-4;
maxiter = 2e2;
Theta_init = zeros(nr, 2*nc);

%[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
[Theta_hat, ~] = MVAPG_MCP(y, X_new, type, Clambda, tol, maxiter, Theta_init);

disp(sum(sum((Theta_hat(:,1:nc) - Theta_star(:,:,1)).^2)));


%% MatLassoSCP
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e3);
APG_opts = struct(...
    'type', type,...
    'Clambda', 0.15,...
    'tol', 1e-4,...
    'maxiter', 1e2,...
    'Theta_init', zeros(nr, 2*nc));
[Theta_Delta_hat, tau_hat, obj_path, Delta_path] = MatLassoSCP(y, X, threshold_var, 0.2, [0,1], 50, APG_opts);
figure;
plot(obj_path);
figure;
plot(Delta_path);


%% MatLassoMCP
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 5e3);
Clambda_base  = 0.15;
window_length = 0.25;
num_windows   = 10;
cutoff        = 0.2;

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 1e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 1e2);
SCP_args_2 = struct(...
    "kappa", 0.1,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.15,...
    "tol", 1e-4,...
    "maxiter", 1e2);
    
MCP_opts   = struct();


[post_Theta_hat, post_tau_hat, post_rank, MCP_outInfo] = ...
    MatLassoMCP(y, X, threshold_var, Clambda_base,...
    window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);






