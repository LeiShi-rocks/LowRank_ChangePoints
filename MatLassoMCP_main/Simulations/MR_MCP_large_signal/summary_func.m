% summary of the results
%% MR MCP large
clear;clc;
load('MR_MCP_large_signal.mat');


% record{1, iter} = post_Theta_hat;
% record{2, iter} = post_tau_hat;
% record{3, iter} = post_rank;
% record{4, iter} = MCP_outInfo;
% record{5, iter} = Theta_star;

% MCP_outInfo.tau_hat_path  = tau_hat_path;
% MCP_outInfo.best_obj_path = best_obj_path;
% MCP_outInfo.best_Delta_Fnormsq_path = best_Delta_Fnormsq_path;
% MCP_outInfo.pre_tau_hat   = pre_tau_hat;

tau_star     = [0.25, 0.50, 0.75];
reports_vec  = zeros(100, 1);
reports_avg  = zeros(10,1); % 10 criteria, 2 settings (pre and post)
reports_std  = zeros(10,1);


% number of pre selected breaks
for iter = 1:100
    reports_vec(iter) = length(record{4, iter}.pre_tau_hat);
end
reports_avg(1, 1) = mean(reports_vec);
reports_std(1, 1) = std(reports_vec);


% OE for pre selected
for iter = 1:100
    contrast_tab = abs(tau_star' - record{4, iter}.pre_tau_hat);
    reports_vec(iter) = max(min(contrast_tab, [], 2));
end
reports_avg(2, 1) = mean(reports_vec);
reports_std(2, 1) = std(reports_vec);


% UE for pre selected
for iter = 1:100
    contrast_tab = abs(tau_star' - record{4, iter}.pre_tau_hat);
    reports_vec(iter) = max(min(contrast_tab, [], 1));
end
reports_avg(3, 1) = mean(reports_vec);
reports_std(3, 1) = std(reports_vec);



% number of post selected breaks
for iter = 1:100
    reports_vec(iter) = length(record{2, iter});
end
reports_avg(4, 1) = mean(reports_vec);
reports_std(4, 1) = std(reports_vec);



% OE for post selected
for iter = 1:100
    contrast_tab = abs(tau_star' - record{2, iter});
    reports_vec(iter) = max(min(contrast_tab, [], 2));
end
reports_avg(5, 1) = mean(reports_vec);
reports_std(5, 1) = std(reports_vec);



% UE for post selected
for iter = 1:100
    contrast_tab = abs(tau_star' - record{2, iter});
    reports_vec(iter) = max(min(contrast_tab, [], 1));
end
reports_avg(6, 1) = mean(reports_vec);
reports_std(6, 1) = std(reports_vec);



% OE for matrices in Fnorm squared
for iter = 1:100
    Theta_star = record{5, iter}; 
    num_star = size(Theta_star, 3);
    post_Theta_hat = record{1, iter};
    num_hat  = size(post_Theta_hat, 3);
    contrast_tab = zeros(num_star, num_hat);
    for ind_star = 1:num_star
        for ind_hat = 1:num_hat
            contrast_tab(ind_star, ind_hat) = sum(sum((Theta_star(:,:,ind_star) - post_Theta_hat(:,:,ind_hat)).^2));
        end
    end
    reports_vec(iter) = max(min(contrast_tab, [], 2));
end
reports_avg(7, 1) = mean(reports_vec);
reports_std(7, 1) = std(reports_vec);


% UE for matrices in Fnorm squared
for iter = 1:100
    Theta_star = record{5, iter}; 
    num_star = size(Theta_star, 3);
    post_Theta_hat = record{1, iter};
    num_hat  = size(post_Theta_hat, 3);
    contrast_tab = zeros(num_star, num_hat);
    for ind_star = 1:num_star
        for ind_hat = 1:num_hat
            contrast_tab(ind_star, ind_hat) = sum(sum((Theta_star(:,:,ind_star) - post_Theta_hat(:,:,ind_hat)).^2));
        end
    end
    reports_vec(iter) = max(min(contrast_tab, [], 1));
end
reports_avg(8, 1) = mean(reports_vec);
reports_std(8, 1) = std(reports_vec);


% max rank
for iter = 1:100
    reports_vec(iter) = max(record{3, iter});
end
reports_avg(9, 1) = mean(reports_vec);
reports_std(9, 1) = std(reports_vec);


% min rank
for iter = 1:100
    reports_vec(iter) = min(record{3, iter});
end
reports_avg(10, 1) = mean(reports_vec);
reports_std(10, 1) = std(reports_vec);


reports_avg_tab = [reports_avg(1), reports_avg(2), reports_avg(3), 0, 0, 0, 0;...
               reports_avg(4), reports_avg(5), reports_avg(6), reports_avg(7), reports_avg(8), reports_avg(9), reports_avg(10)];

reports_std_tab = [reports_std(1), reports_std(2), reports_std(3), 0, 0, 0, 0;...
               reports_std(4), reports_std(5), reports_std(6), reports_std(7), reports_std(8), reports_std(9), reports_std(10)];


disp([fancy_num2str(reports_avg_tab(:,1), reports_std_tab(:,1), 2, 2, 1e-2), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,2), reports_std_tab(:,2), 3, 3, 1e-4), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,3), reports_std_tab(:,3), 3, 3, 1e-4), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,4), reports_std_tab(:,4), 3, 3, 1e-4), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,5), reports_std_tab(:,5), 3, 3, 1e-4), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,6), reports_std_tab(:,6), 2, 2, 1e-3), ...
    repmat('  ', [2,1]),...
    fancy_num2str(reports_avg_tab(:,7), reports_std_tab(:,7), 2, 2, 1e-3)...
    ]);



