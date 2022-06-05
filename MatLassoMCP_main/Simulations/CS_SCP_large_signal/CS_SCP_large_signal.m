%% varying N
clear;clc;
fprintf('Running: CS_SCP_LARGE_SIGNAL, varying N... \n\n')

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
record_N = cell(4, 7, 3, num_iter);
% 4 methods: ours, LASSO, Oracle, NC
% Fnormsq_1, Fnormsq_2, tau_hat, obj_path, Delta_path, Theta_hat,
% Theta_star(in first round of running)

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
            
            APG_opts = struct(...
                'type', type,...
                'Clambda', 0.30,...
                'tol', 1e-4,...
                'maxiter', 2e2,...
                'Theta_init', zeros(2*nr, nc));
            [Theta_Delta_hat, tau_hat, obj_path, Delta_path] =...
                MatLassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_N{1, 1, N_ind, iter}  = Fnormsq_1;
            record_N{1, 2, N_ind, iter}  = Fnormsq_2;
            record_N{1, 3, N_ind, iter}  = tau_hat;
            record_N{1, 4, N_ind, iter}  = obj_path;
            record_N{1, 5, N_ind, iter}  = Delta_path;
            record_N{1, 6, N_ind, iter}  = Theta_Delta_hat;
            record_N{1, 7, N_ind, iter}  = Theta_star;
            
            % output
            fprintf(' N_ind: %d, Iteration: %d, MatLassoSCP \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
        end
        fprintf('\n');
        
        
        % solve LASSO
        if running_flag(2)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            
            APG_opts = struct(...
                'type', type,...
                'Clambda', 0.30,...
                'tol', 1e-4,...
                'maxiter', 2e2,...
                'Theta_init', zeros(2*nr, nc));
            
            [Theta_Delta_hat, tau_hat, obj_path, Delta_path] =...
                LassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_N{2, 1, N_ind, iter}  = Fnormsq_1;
            record_N{2, 2, N_ind, iter}  = Fnormsq_2;
            record_N{2, 3, N_ind, iter}  = tau_hat;
            record_N{2, 4, N_ind, iter}  = obj_path;
            record_N{2, 5, N_ind, iter}  = Delta_path;
            record_N{2, 6, N_ind, iter}  = Theta_Delta_hat;
            
            % output
            fprintf(' N_ind: %d, Iteration: %d, LassoSCP \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
        end
        fprintf('\n');
        
        
        % solve Oracle
        if running_flag(3)
            X_new = zeros(2*nr, nc, N);
            for i = 1:N
                X_new(:,:,i) = [X(:,:,i); X(:,:,i) .* (threshold_var(i) > 0.5)];
            end
            
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.30;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(2*nr, nc);
            
            [Theta_Delta_hat, ~] = MMAPG_MCP(y, X_new, type, Clambda, tol, maxiter, Theta_init);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_N{3, 1, N_ind, iter}  = Fnormsq_1;
            record_N{3, 2, N_ind, iter}  = Fnormsq_2;
            record_N{3, 3, N_ind, iter}  = 0;
            record_N{3, 4, N_ind, iter}  = 0;
            record_N{3, 5, N_ind, iter}  = 0;
            record_N{3, 6, N_ind, iter}  = Theta_Delta_hat;
            
            % output
            fprintf(' N_ind: %d, Iteration: %d, Oracle \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
        end
        
        fprintf('\n');
        
        
        % solve Non change point
        if running_flag(4)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.30;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            [Theta_hat, ~] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat - Theta_star(:,:,2)).^2));
            
            record_N{4, 1, N_ind, iter}  = Fnormsq_1;
            record_N{4, 2, N_ind, iter}  = Fnormsq_2;
            record_N{4, 3, N_ind, iter}  = 0;
            record_N{4, 4, N_ind, iter}  = 0;
            record_N{4, 5, N_ind, iter}  = 0;
            record_N{4, 6, N_ind, iter}  = Theta_hat;
            
            fprintf(' N_ind: %d, Iteration: %d, Non Change \n', N_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
        end
        fprintf('\n');
        
    end
end

if save_flag
    save('record_N_CS_SCP_large_signal.mat', 'record_N');
end


%% varying dimension
clear;clc;

fprintf('Running: CS_SCP_LARGE_SIGNAL, varying dimension... \n\n')

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

running_flag = [1,1,1,1];

num_iter = 100;
record_dim = cell(4, 7, 3, num_iter);
% 4 methods: ours, LASSO, Oracle, NC
% Fnormsq_1, Fnormsq_2, tau_hat, obj_path, Delta_path

save_flag = 1;

rng(2022);

for dim_ind = 1:length(dim_Cand)
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
                'Lf', 5e4);
            
            APG_opts = struct(...
                'type', type,...
                'Clambda', 0.3,...
                'tol', 1e-4,...
                'maxiter', 2e2,...
                'Theta_init', zeros(2*nr, nc));
            [Theta_Delta_hat, tau_hat, obj_path, Delta_path] =...
                MatLassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_dim{1, 1, dim_ind, iter}  = Fnormsq_1;
            record_dim{1, 2, dim_ind, iter}  = Fnormsq_2;
            record_dim{1, 3, dim_ind, iter}  = tau_hat;
            record_dim{1, 4, dim_ind, iter}  = obj_path;
            record_dim{1, 5, dim_ind, iter}  = Delta_path;
            record_dim{1, 6, dim_ind, iter}  = Theta_Delta_hat;
            record_dim{1, 7, dim_ind, iter}  = Theta_star;
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, MatLassoSCP \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
        end
        fprintf('\n');
        
        
        % solve LASSO
        if running_flag(2)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            
            APG_opts = struct(...
                'type', type,...
                'Clambda', 0.30,...
                'tol', 1e-4,...
                'maxiter', 2e2,...
                'Theta_init', zeros(2*nr, nc));
            
            [Theta_Delta_hat, tau_hat, obj_path, Delta_path] =...
                LassoSCP(y, X, outInfo.threshold_var, 0.2, [0,1], 50, APG_opts);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_dim{2, 1, dim_ind, iter}  = Fnormsq_1;
            record_dim{2, 2, dim_ind, iter}  = Fnormsq_2;
            record_dim{2, 3, dim_ind, iter}  = tau_hat;
            record_dim{2, 4, dim_ind, iter}  = obj_path;
            record_dim{2, 5, dim_ind, iter}  = Delta_path;
            record_dim{2, 6, dim_ind, iter}  = Theta_Delta_hat;
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, LassoSCP \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d, tau_hat: %d \n', Fnormsq_1, Fnormsq_2, tau_hat);
        end
        fprintf('\n');
        
        % solve Oracle
        if running_flag(3)
            X_new = zeros(2*nr, nc, N);
            for i = 1:N
                X_new(:,:,i) = [X(:,:,i); X(:,:,i) .* (threshold_var(i) > 0.5)];
            end
            
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.30;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(2*nr, nc);
            
            [Theta_Delta_hat, ~] = MMAPG_MCP(y, X_new, type, Clambda, tol, maxiter, Theta_init);
            
            % store information
            Fnormsq_1 = sum(sum((Theta_Delta_hat(1:nr,:) - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_Delta_hat(1:nr,:) + Theta_Delta_hat((nr+1):(2*nr),:) - Theta_star(:,:,2)).^2));
            
            record_dim{3, 1, dim_ind, iter}  = Fnormsq_1;
            record_dim{3, 2, dim_ind, iter}  = Fnormsq_2;
            record_dim{3, 3, dim_ind, iter}  = 0;
            record_dim{3, 4, dim_ind, iter}  = 0;
            record_dim{3, 5, dim_ind, iter}  = 0;
            record_dim{3, 6, dim_ind, iter}  = Theta_Delta_hat;
            
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, Oracle \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
        end
        fprintf('\n');
        
        
        % solve Non change point
        if running_flag(4)
            type = struct(...
                'name', 'L2',...
                'eta', 0.8,...
                'Lf', 5e4);
            Clambda = 0.30;
            tol = 1e-4;
            maxiter = 2e2;
            Theta_init = zeros(nr, nc);
            
            %[Theta_hat, rank] = MVAPG(y, X, type, lambda, tol, maxiter, Theta_init);
            [Theta_hat, ~] = MMAPG_MCP(y, X, type, Clambda, tol, maxiter, Theta_init);
            
            Fnormsq_1 = sum(sum((Theta_hat - Theta_star(:,:,1)).^2));
            Fnormsq_2 = sum(sum((Theta_hat - Theta_star(:,:,2)).^2));
            
            % store information
            record_dim{4, 1, dim_ind, iter}  = Fnormsq_1;
            record_dim{4, 2, dim_ind, iter}  = Fnormsq_2;
            record_dim{4, 3, dim_ind, iter}  = 0;
            record_dim{4, 4, dim_ind, iter}  = 0;
            record_dim{4, 5, dim_ind, iter}  = 0;
            record_dim{4, 6, dim_ind, iter}  = Theta_hat;
            
            % output
            fprintf(' dim_ind: %d, Iteration: %d, Non Change \n', dim_ind, iter);
            fprintf(' Fnormsq_1: %d, Fnormsq_2: %d \n', Fnormsq_1, Fnormsq_2);
        end
        fprintf('\n');
        
    end
end

if save_flag
    save('record_dim_CS_SCP_large_signal.mat', 'record_dim');
end
















