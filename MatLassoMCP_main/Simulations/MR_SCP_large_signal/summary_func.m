% summary of the results

%% MR SCP large
clear;clc;
load('record_N_refit_MR_SCP_large_signal.mat'); % nr = nc = 50;
% record_N = cell(4, 7, 3, num_iter);
% 4 methods: ours, LASSO, Oracle, NC
% Fnormsq_1, Fnormsq_2, tau_hat, obj_path, Delta_path, Theta_hat,
% Theta_star(in first round of running)

reports_vec  = zeros(100, 1);
reports_avg  = zeros(4, 7, 3); % 4 methods, 7 criteria, 3 N
reports_std  = zeros(4, 7, 3);

for N_ind = 1:3
    for method_ind = 1:4
        fprintf('N_ind: %d, method_ind: %d \n\n', N_ind, method_ind);
        % tau
        for iter = 1:100
            reports_vec(iter) = abs(record_N_refit{method_ind, 3, N_ind, iter} - 0.5);
        end
        reports_avg(method_ind, 1, N_ind) = mean(reports_vec);
        reports_std(method_ind, 1, N_ind) = std(reports_vec);
        
        % Fnorm_1^2
        for iter = 1:100
            reports_vec(iter) = record_N_refit{method_ind, 1, N_ind, iter}(1);
        end
        reports_avg(method_ind, 2, N_ind) = mean(reports_vec);
        reports_std(method_ind, 2, N_ind) = std(reports_vec);
        
        % *norm_1
        for iter = 1:100
            Theta_star = record_N_refit{1, 7, N_ind, iter};
            Theta_hat  = record_N_refit{method_ind, 6, N_ind, iter};
            
            Theta_hat_1 = Theta_hat(:, 1:50);
            Theta_hat_2 = Theta_hat(:, 51:100);
            
            reports_vec(iter) = sum(svd(Theta_hat_1 - Theta_star(:,:,1)));
        end
        reports_avg(method_ind, 3, N_ind) = mean(reports_vec);
        reports_std(method_ind, 3, N_ind) = std(reports_vec);
        
        % rank_1
        for iter = 1:100
            reports_vec(iter) = record_N_refit{method_ind, 1, N_ind, iter}(2);
        end
        reports_avg(method_ind, 4, N_ind) = mean(reports_vec);
        reports_std(method_ind, 4, N_ind) = std(reports_vec);
        
        % Fnorm_2^2
        for iter = 1:100
            reports_vec(iter) = record_N_refit{method_ind, 2, N_ind, iter}(1);
        end
        reports_avg(method_ind, 5, N_ind) = mean(reports_vec);
        reports_std(method_ind, 5, N_ind) = std(reports_vec);
        
        % *norm_2
        for iter = 1:100
            Theta_star = record_N_refit{1, 7, N_ind, iter};
            Theta_hat  = record_N_refit{method_ind, 6, N_ind, iter};
            Theta_hat_1 = Theta_hat(:, 1:50);
            Theta_hat_2 = Theta_hat(:, 51:100);
            
            reports_vec(iter) = sum(svd(Theta_hat_2 - Theta_star(:,:,2)));
        end
        reports_avg(method_ind, 6, N_ind) = mean(reports_vec);
        reports_std(method_ind, 6, N_ind) = std(reports_vec);
        
        % rank_2
        for iter = 1:100
            reports_vec(iter) = record_N_refit{method_ind, 2, N_ind, iter}(2);
        end
        reports_avg(method_ind, 7, N_ind) = mean(reports_vec);
        reports_std(method_ind, 7, N_ind) = std(reports_vec);
    end
    
    reports_avg(reports_avg < 1e-8) = 0;
    reports_std(reports_std < 1e-8) = 0;
    
    disp([
        fancy_num2str(reports_avg(:,1,N_ind), reports_std(:,1,N_ind), 4, 4, 1e-4), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,2,N_ind), reports_std(:,2,N_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,3,N_ind), reports_std(:,3,N_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,4,N_ind), reports_std(:,4,N_ind), 2, 2, 1e-2), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,5,N_ind), reports_std(:,5,N_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,6,N_ind), reports_std(:,6,N_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,7,N_ind), reports_std(:,7,N_ind), 2, 2, 1e-2), ...
        repmat('  ', [4,1]), ...
    ]);
    
end

%% plot: boxplot_N

plot_vec_ML = zeros(100,3);
plot_vec_VL = zeros(100,3);
% boxplot for change points
figure;
for N_ind = 1:3
    for iter = 1:100
        plot_vec_ML(iter, N_ind) = record_N_refit{1, 3, N_ind, iter};
    end
end

for N_ind = 1:3
    for iter = 1:100
        plot_vec_VL(iter, N_ind) = record_N_refit{2, 3, N_ind, iter};
    end
end

labels = {'ML, N=500', 'ML, N=1000', 'ML, N=2000', 'VL, N=500', 'VL, N=1000', 'VL, N=2000'};
boxplot([plot_vec_ML, plot_vec_VL], 'Labels', labels);
xlabel('Method and sample size', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('Estimated change point', 'FontWeight', 'bold', 'FontSize', 14);
%boxplot(plot_vec_ML, 'Labels', labels(1:3));

%% plot: obj trajectory
%figure;

figure;
plot_vec_path = zeros(100,100);
for iter = 1:100
    plot_vec_path(:,iter) = record_N_refit{1, 4, 3, iter}';
end

plot(0.01*(1:100), plot_vec_path, 'Color', [0.6 0.6 0.6]);
hold on;
plot(0.01*(1:100), mean(plot_vec_path, 2), 'Color', 'red', 'LineWidth', 3);
xlabel("Position \tau", 'FontWeight', 'bold', 'FontSize', 14);
ylabel("Objective value", 'FontWeight', 'bold', 'FontSize', 14);


%% MR SCP large
clear;clc;
load('record_dim_refit_MR_SCP_large_signal.mat'); % nr = nc = 50;
% record_N = cell(4, 7, 3, num_iter);
% 4 methods: ours, LASSO, Oracle, NC
% Fnormsq_1, Fnormsq_2, tau_hat, obj_path, Delta_path, Theta_hat,
% Theta_star(in first round of running)

reports_vec  =  zeros(100, 1);
reports_avg  =  zeros(4, 7, 3); % 4 methods, 7 criteria, 3 dim
reports_std  =  zeros(4, 7, 3);

dim_Cand = [25, 50, 75];

for dim_ind = 1:3
    for method_ind = 1:4
        fprintf('dim_ind: %d, method_ind: %d \n\n', dim_ind, method_ind);
        % tau
        for iter = 1:100
            reports_vec(iter) = abs(record_dim_refit{method_ind, 3, dim_ind, iter} - 0.5);
        end
        reports_avg(method_ind, 1, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 1, dim_ind) = std(reports_vec);
        
        % Fnorm_1^2
        for iter = 1:100
            reports_vec(iter) = record_dim_refit{method_ind, 1, dim_ind, iter}(1);
        end
        reports_avg(method_ind, 2, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 2, dim_ind) = std(reports_vec);
        
        % *norm_1
        for iter = 1:100
            Theta_star = record_dim_refit{1, 7, dim_ind, iter};
            Theta_hat  = record_dim_refit{method_ind, 6, dim_ind, iter};
            
            Theta_hat_1 = Theta_hat(:, 1:dim_Cand(dim_ind));
            Theta_hat_2 = Theta_hat(:, (dim_Cand(dim_ind)+1):end);
            
            reports_vec(iter) = sum(svd(Theta_hat_1 - Theta_star(:,:,1)));
        end
        reports_avg(method_ind, 3, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 3, dim_ind) = std(reports_vec);
        
        % rank_1
        for iter = 1:100
            reports_vec(iter) = record_dim_refit{method_ind, 1, dim_ind, iter}(2);
        end
        reports_avg(method_ind, 4, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 4, dim_ind) = std(reports_vec);
        
        % Fnorm_2^2
        for iter = 1:100
            reports_vec(iter) = record_dim_refit{method_ind, 2, dim_ind, iter}(1);
        end
        reports_avg(method_ind, 5, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 5, dim_ind) = std(reports_vec);
        
        % *norm_2
        for iter = 1:100
            Theta_star = record_dim_refit{1, 7, dim_ind, iter};
            Theta_hat  = record_dim_refit{method_ind, 6, dim_ind, iter};
            Theta_hat_1 = Theta_hat(:, 1:dim_Cand(dim_ind));
            Theta_hat_2 = Theta_hat(:, (dim_Cand(dim_ind)+1):end);
            
            reports_vec(iter) = sum(svd(Theta_hat_2 - Theta_star(:,:,2)));
        end
        reports_avg(method_ind, 6, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 6, dim_ind) = std(reports_vec);
        
        % rank_2
        for iter = 1:100
            reports_vec(iter) = record_dim_refit{method_ind, 2, dim_ind, iter}(2);
        end
        reports_avg(method_ind, 7, dim_ind) = mean(reports_vec);
        reports_std(method_ind, 7, dim_ind) = std(reports_vec);
    end
    
    reports_avg(reports_avg < 1e-8) = 0;
    reports_std(reports_std < 1e-8) = 0;
    
    disp([
        fancy_num2str(reports_avg(:,1,dim_ind), reports_std(:,1,dim_ind), 4, 4, 1e-4), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,2,dim_ind), reports_std(:,2,dim_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,3,dim_ind), reports_std(:,3,dim_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,4,dim_ind), reports_std(:,4,dim_ind), 2, 2, 1e-2), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,5,dim_ind), reports_std(:,5,dim_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,6,dim_ind), reports_std(:,6,dim_ind), 3, 3, 1e-3), ...
        repmat('  ', [4,1]), ...
        fancy_num2str(reports_avg(:,7,dim_ind), reports_std(:,7,dim_ind), 2, 2, 1e-2), ...
        repmat('  ', [4,1]), ...
    ]);
end

%% plot: boxplot_dim

plot_vec_ML = zeros(100,3);
plot_vec_VL = zeros(100,3);
% boxplot for change points
figure;
for dim_ind = 1:3
    for iter = 1:100
        plot_vec_ML(iter, dim_ind) = record_dim_refit{1, 3, dim_ind, iter};
    end
end

for dim_ind = 1:3
    for iter = 1:100
        plot_vec_VL(iter, dim_ind) = record_dim_refit{2, 3, dim_ind, iter};
    end
end

labels = {'ML, m=25', 'ML, m=50', 'ML, m=75', 'VL, m=25', 'VL, m=50', 'VL, m=75'};
boxplot([plot_vec_ML, plot_vec_VL], 'Labels', labels);
xlabel('Method and sample size', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('Estimated change point', 'FontWeight', 'bold', 'FontSize', 14);
%boxplot(plot_vec_ML, 'Labels', labels(1:3));

%% plot: obj trajectory
%figure;

figure;
plot_vec_path = zeros(100,100);
for iter = 1:100
    plot_vec_path(:,iter) = record_dim_refit{1, 4, 3, iter}';
end

plot(0.01*(1:100), plot_vec_path, 'Color', [0.6 0.6 0.6]);
hold on;
plot(0.01*(1:100), mean(plot_vec_path, 2), 'Color', 'red', 'LineWidth', 3);
xlabel("Position \tau", 'FontWeight', 'bold', 'FontSize', 14);
ylabel("Objective value", 'FontWeight', 'bold', 'FontSize', 14);