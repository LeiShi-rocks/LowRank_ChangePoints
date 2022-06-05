%% varying N
clear;clc;
fprintf('Running: CS_SCP_LARGE_SIGNAL, varying N... \n\n')
load('record_N_CS_SCP_large_signal.mat');
%load('record_N_refit_MR_SCP_large_signal.mat');
nr = 40;
nc = 40;

N_Cand = [1.5e3, 2e3, 2.5e3];
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

running_flag = [1,1,1,1];

num_iter = 100;
record_N_refit = record_N;
% 4 methods: ours, LASSO, Oracle, NC
% Fnormsq_1, Fnormsq_2, tau_hat, obj_path, Delta_path, Theta_Delta_hat,
% Theta_star

save_flag = 1;

rng(2022);

for N_ind = 1:length(N_Cand)
    N = N_Cand(N_ind);
    for iter = 1:num_iter
        % data generation
        [y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
        Theta_star = outInfo.Theta_star;
        threshold_var = outInfo.threshold_var;
        
        % solve MatLassoSCP
        if running_flag(1)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = record_N{1, 3, N_ind, iter};
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg),  X(:,:,first_seg),  type, Clambda, tol, maxiter, Theta_init);
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            record_N_refit{1, 1, N_ind, iter}  = [Fnormsq_1, rank1];
            record_N_refit{1, 2, N_ind, iter}  = [Fnormsq_2, rank2];
            record_N_refit{1, 6, N_ind, iter}  = [Theta_hat_1, Theta_hat_2];
            
            fprintf(' N_ind: %d, Iteration: %d, MatLasso \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
        end
        fprintf('\n');
        
        
        % solve LASSO
        if running_flag(2)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.05;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = record_N{2, 3, N_ind, iter};
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg),  X(:,:,first_seg),  type, Clambda, tol, maxiter, Theta_init, 'l1');
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init, 'l1');
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            
            record_N_refit{2, 1, N_ind, iter}  =  [Fnormsq_1, rank1];
            record_N_refit{2, 2, N_ind, iter}  =  [Fnormsq_2, rank2];
            record_N_refit{2, 6, N_ind, iter}  =  [Theta_hat_1, Theta_hat_2];
            
            % output
            fprintf(' N_ind: %d, Iteration: %d, LassoSCP \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
        end
        fprintf('\n');
        
        
        % solve Oracle
        if running_flag(3)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = 0.5;
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg), X(:,:,first_seg), type, Clambda, tol, maxiter, Theta_init);
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            record_N_refit{3, 1, N_ind, iter}  =  [Fnormsq_1, rank1];
            record_N_refit{3, 2, N_ind, iter}  =  [Fnormsq_2, rank2];
            record_N_refit{3, 6, N_ind, iter}  =  [Theta_hat_1, Theta_hat_2];
            
            fprintf(' N_ind: %d, Iteration: %d, Oracle \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
        end
        fprintf('\n');
        
        
        % solve Non change point
        if running_flag(4)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            [Theta_hat, rank] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat - Theta_star(:,:,2)).^2));
            
            record_N_refit{4, 1, N_ind, iter}  = [Fnormsq_1, rank];
            record_N_refit{4, 2, N_ind, iter}  = [Fnormsq_2, rank];
            record_N_refit{4, 6, N_ind, iter}  = [Theta_hat, Theta_hat];
            
            fprintf(' N_ind: %d, Iteration: %d, Non Change \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
            fprintf(' rank1: %d, rank2: %d \n', rank, rank);
        end
        fprintf('\n');
        
    end
end

if save_flag
    save('record_N_refit_CS_SCP_large_signal.mat', 'record_N_refit');
end


%% varying dimension
clear;clc;
load('record_dim_CS_SCP_large_signal.mat');
fprintf('Running: CS_SCP_LARGE_SIGNAL, varying dimension... \n\n')
%load('record_dim_refit_MR_SCP_small_signal.mat');

dim_Cand = [20, 35, 50];

N_Cand = 10*5*dim_Cand;
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

num_iter = 100;

running_flag = [1,1,1,1];
record_dim_refit = record_dim;
% 4 methods: ours, LASSO, Oracle, NC
% [Fnormsq_1, rank1], [Fnormsq_2, rank2], tau_hat, obj_path, Delta_path

save_flag = 1;

rng(2022);

for dim_ind = 1:length(N_Cand)
    nr = dim_Cand(dim_ind);
    nc = nr;
    N = N_Cand(dim_ind);
    for iter = 1:num_iter
        % data generation
        [y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
        Theta_star = outInfo.Theta_star;
        threshold_var = outInfo.threshold_var;
        
        % solve MatLassoSCP
        if running_flag(1)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 1e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = record_dim{1, 3, dim_ind, iter};
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg), X(:,:,first_seg), type, Clambda, tol, maxiter, Theta_init);
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            record_dim_refit{1, 1, dim_ind, iter}  = [Fnormsq_1, rank1];
            record_dim_refit{1, 2, dim_ind, iter}  = [Fnormsq_2, rank2];
            record_dim_refit{1, 6, dim_ind, iter}  = [Theta_hat_1, Theta_hat_2];
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, MatLassoSCP \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
           
        end
        fprintf('\n');
        
        
        % solve LASSO
        if running_flag(2)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.05;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = record_dim{2, 3, dim_ind, iter};
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg), X(:,:,first_seg), type, Clambda, tol, maxiter, Theta_init, 'l1');
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init, 'l1');
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            
            record_dim_refit{2, 1, dim_ind, iter}  = [Fnormsq_1, rank1];
            record_dim_refit{2, 2, dim_ind, iter}  = [Fnormsq_2, rank2];
            record_dim_refit{2, 6, dim_ind, iter}  = [Theta_hat_1, Theta_hat_2];
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, LassoSCP \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
            
            
            
        end
        fprintf('\n');
        
        
        
        % solve Oracle
        X_new = zeros(2*nc, N);
        if running_flag(3)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            tau_hat = 0.5;
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            first_seg  = find((1:N)/N < tau_hat);
            second_seg = find((1:N)/N >= tau_hat);
            
            
            [Theta_hat_1, rank1] = MMAPG_MCP(y(first_seg), X(:,:,first_seg), type, Clambda, tol, maxiter, Theta_init);
            [Theta_hat_2, rank2] = MMAPG_MCP(y(second_seg), X(:,:,second_seg), type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat_1 - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat_2 - Theta_star(:,:,2)).^2));
            
            record_dim_refit{3, 1, dim_ind, iter}  = [Fnormsq_1, rank1];
            record_dim_refit{3, 2, dim_ind, iter}  = [Fnormsq_2, rank2];
            record_dim_refit{3, 6, dim_ind, iter}  = [Theta_hat_1, Theta_hat_2];
                   
            % output
            fprintf(' dim_ind: %d, Iteration: %d, Oracle \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
            fprintf(' rank1: %d, rank2: %d \n', rank1, rank2);
        end
        fprintf('\n');
        
        
        % solve Non change point
        if running_flag(4)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.80;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            [Theta_hat, rank] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat - Theta_star(:,:,2)).^2));
            
            % store information
            record_dim_refit{4, 1, dim_ind, iter}  = [Fnormsq_1, rank];
            record_dim_refit{4, 2, dim_ind, iter}  = [Fnormsq_2, rank];
            record_dim_refit{4, 6, dim_ind, iter}  = [Theta_hat, Theta_hat];
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, Non Change \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
            fprintf(' rank1: %d, rank2: %d \n', rank, rank);
        end
        fprintf('\n');
        
    end
end

if save_flag
    save('record_dim_refit_CS_SCP_large_signal.mat', 'record_dim_refit');
end
















