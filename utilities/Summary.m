%% linear regression tuning 
% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/Computation/Record_LR_Tuning.mat')
% Record_LR_Tuning
% 3 methods, 5 indices(loss, sparsity, tuning+computation+total time), niter, 3 noise
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Computation\Record_LR_Tuning.mat')
Result = zeros(3,5,3);
p1 = permute(Record_LR_Tuning, [1 2 4 3]);
for i = 1 : 100
    Result = Result + p1(:,:,:,i)/100;
end

% Result(:,1,:) = log(Result(:,1,:));

Resultsd = std(Record_LR_Tuning, 0, 3);
% disp(num2str(Result(:, :, noiseInd)))

% for noiseInd = 1 : 3
% for nrow = 1 : 3
%     rowRec = [];
%     for ncol = [1 3 4 5]
%         rowRec = [rowRec, '  ', num2str(Result(nrow,ncol,noiseInd), '%.2f'),...
%             '(', num2str(Resultsd(nrow,ncol,noiseInd), '%.2f'),')'];
%     end
%     disp(rowRec);
% end
% end

for noiseInd = 1 : 3
for nrow = 1 : 3
    rowRec = [];
    for ncol = [1 3 4 5]
        rowRec = [rowRec, '  ', num2str(Result(nrow,ncol,noiseInd), '%.2f')];
    end
    disp(rowRec);
end
end


%% compressed sensing tuning
% seven methods(APGL-L2, L2, Robust-L2, L1, Huber, Wilcoxon-cv, Wilcoxon-pivotal),
% 5 indices(CVlambda, CVtau for Fan2016, tuning time, solving time, total time, loss), 
% 3 noise, niter;

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/Computation/Record_CS_small_d40_tuning.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Computation\Record_CS_small_d40_tuning.mat')
PreResult = mean(Record_CS_small_d40_tuning, 4);
PreResult = PreResult(:,[1 2 6 3 4 5],:);
Result = PreResult([3 6 7], 3:6, :); 
% Result(:,1,:) = log(Result(:,1,:)*1600)/2;
Result(:,1,:) = Result(:,1,:)*1600;


PreResultsd = std(Record_CS_small_d40_tuning, 0, 4);
PreResultsd = PreResultsd(:,[1 2 6 3 4 5],:);
Resultsd = PreResultsd([3 6 7], 3:6, :); 


% disp(num2str(Result(:,:,1), 4))
% disp(num2str(Result(:,:,2), 4))
% disp(num2str(Result(:,:,3), 4))
% 
% disp(num2str(Resultsd(:,:,1), 4))
% disp(num2str(Resultsd(:,:,2), 4))
% disp(num2str(Resultsd(:,:,3), 4))

% for noiseInd = 1 : 3
% for nrow = 1 : 3
%     rowRec = [];
%     for ncol = 1 : 4
%         rowRec = [rowRec, '  ', num2str(Result(nrow,ncol,noiseInd),'%.2e'),...
%             '(', num2str(Resultsd(nrow,ncol,noiseInd), '%.2e'),')'];
%     end
%     disp(rowRec);
% end
% end

for noiseInd = 1 : 3
for nrow = 1 : 3
    rowRec = [];
    for ncol = 1 : 4
        rowRec = [rowRec, '  ', num2str(Result(nrow,ncol,noiseInd),'%.2e')];
    end
    disp(rowRec);
end
end

%% multivariate regression d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_d40.mat')
% Record_MR_small_d40
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_d40.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_MR_small_d40_ardesign0.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(nr/2, 2*nr, 6));

Result = zeros(6,3,3,6);
p1 = permute(Record_MR_small_d40_ardesign, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([0.3 1.1]);
xlim([2.5 4.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
xlim([2.5 4.9]);
ylim([-4, 8.3]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
xlim([2.5 4.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;


% disp(num2str(log(p2(:,[2 3 4 6],noiseInd,1)'*nr*nc)/2))


%% multivariate regression d = 40 under AR design

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_d40.mat')
% Record_MR_small_d40
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_MR_small_d40_ardesign08.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(nr/2, 2*nr, 6));

Result = zeros(6,3,3,6);
p1 = permute(Record_MR_small_d40_ardesign, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([0.5 1.1]);xlim([2.5 4.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-2,8.5]);xlim([2.5 4.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-1 7]);xlim([2.5 4.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;


% disp(num2str(log(p2(:,[2 3 4 6],noiseInd,1)'*nr*nc)/2))

% heatmap the covariance matrix

map = repmat((1:-0.05:0)', [1,3]);

figure; 

covMat = eye(40);
imagesc(covMat); % Display correlation matrix as an image
%title('Heatmap of the identity covariance matrix', 'FontSize', 10); % set title
colormap(map); % Choose jet or any other color scheme
colorbar; % 



figure; 

rows = repmat((1:nc)', [1, nc]);
cols = repmat((1:nc) , [nc, 1]);
Amatrix = 0.8*ones(nc);
covMat = Amatrix.^(abs(rows-cols));

imagesc(covMat); % Display correlation matrix as an image
%title('Heatmap of the autoregressive covariance matrix(a=0.5)', 'FontSize', 10); % set title
colormap(map); % Choose jet or any other color scheme
colorbar; % 


%% multivariate regression with t(n) design under d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_tn_d40.mat');
% Record_MR_small_tn_d40
% method, record(loss, rank, time), niter, noise(t(3*1:6))

% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_tn_d40.mat');
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\lowX.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;


Result = zeros(6,6,3); % df, method, indices
p1 = permute(Record_MR_small_Xtn_d40, [4 1 2 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,i)/niter;
end

% p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot([2.1 2.5 3 3.5 4 4.5], log(Result(:,[1 3 4 6],1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
% ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

% A graph of the t_n distributions
figure;

tDensity = zeros(7, 601);
for i = 1 : 6
    tDensity(i, :) = tpdf(-3:0.01:3, 3*i);
end
tDensity(7, :) = normpdf(-3:0.01:3, 0, 1);

p = plot(-3:0.01:3, tDensity([1 3 5 7], :)', 'LineWidth', 2.5);
legend({'t(3)', 't(9)', 't(12)', 'N(0,1)'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([0 0.47]);
xlabel('x', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Density p(x)', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'LineStyle', '-');
set(p(2), 'LineStyle', '--');
set(p(3), 'LineStyle', '-.');
set(p(4), 'LineStyle', ':');
axis square;
% figure;
% p = plot(log(N_Cand), log(p2(:,[1 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
% legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
%     'FontSize', 16, 'Location', 'Northwest');
% ylim([-5 6]);
% xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
% ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
% ax = gca;
% set(ax, 'FontSize', 13, 'FontWeight', 'bold');
% set(p(1), 'Marker', 'o', 'LineStyle', '-');
% set(p(2), 'Marker', '*', 'LineStyle', '--');
% set(p(3), 'Marker', 'x', 'LineStyle', '-.');
% set(p(4), 'Marker', 'square', 'LineStyle', ':');
% axis square;

% figure;
% p = plot(log(N_Cand), log(p2(:,[1 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
% legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
%     'FontSize', 16, 'Location', 'Northwest');
% xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
% ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
% ax = gca;
% set(ax, 'FontSize', 13, 'FontWeight', 'bold');
% set(p(1), 'Marker', 'o', 'LineStyle', '-');
% set(p(2), 'Marker', '*', 'LineStyle', '--');
% set(p(3), 'Marker', 'x', 'LineStyle', '-.');
% set(p(4), 'Marker', 'square', 'LineStyle', ':');
% axis square;


% disp(num2str(log(p2(:,[2 3 4 6],noiseInd,1)'*nr*nc)/2))


%% multivariate regression with t(n) noise under d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_tn_d40.mat');
% Record_MR_small_tn_d40
% method, record(loss, rank, time), niter, noise(t(3*1:6))

% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_tn_d40.mat');
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\lowX.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\lowN.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;


Result = zeros(6,6,3); % df, method, indices
p1 = permute(Record_MR_small_tn_d40, [4 1 2 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,i)/niter;
end

% p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(3*(1:6), log(Result(:,[1 3 4 6],1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
% ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
% xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;


%% multivariate regression with t(n) design under high d

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_tn_d40.mat');
% Record_MR_small_tn_d40
% method, record(loss, rank, time), niter, noise(t(3*1:6))

% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_tn_d40.mat');
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\lowX.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\highX.mat')
niter = 100;
nr = 40;
nc = 80;
% N = 2000;
r = 5;


Result = zeros(6,6,3); % df, method, indices
p1 = permute(Record_MR_small_Xtn_d40, [4 1 2 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,i)/niter;
end

% p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot([2.1 2.5 3 3.5 4 4.5], log(Result(:,[1 3 4 6],1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
% ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;


%% multivariate regression with t(n) noise under high d

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_tn_d40.mat');
% Record_MR_small_tn_d40
% method, record(loss, rank, time), niter, noise(t(3*1:6))

% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_tn_d40.mat');
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\lowX.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\highN.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;


Result = zeros(6,6,3); % df, method, indices
p1 = permute(Record_MR_small_tn_d40, [4 1 2 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,i)/niter;
end

% p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(3*(1:6), log(Result(:,[1 3 4 6],1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
% ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
% xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;




%% multivariate regression d = 80
load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_d80.mat')
% Record_MR_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N

niter = 40;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 2*nr*nc, 6));

Result = zeros(6,3,3,6);
p1 = permute(Record_MR_small_d80, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 

legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
ylim([-2.2, -0.6]);
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],2,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
ylim([-8 6])
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;





%% compressed sensing d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/CompressedSensing/Record_CS_small_d40.mat')
% Record_CS_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\CompressedSensing\Record_CS_small_d40.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_CS_small_d40.mat')
niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 2*nr*nc, 6));

Result = zeros(6,3,3,6);
p1 = permute(Record_CS_small_d40, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],1,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-1.8 -0.4]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],2,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-6 7]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],3,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
% ylim([-6 5]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;



%% compressed sensing d = 80

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/CompressedSensing/Record_CS_small_d80.mat')
% Record_CS_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_CS_small_d80.mat')
niter = 100;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 2*nr*nc, 6));

Result = zeros(6,3,3,6);
p1 = permute(Record_CS_small_d80, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-1 5]); 
xlim([7,10]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
ylim([-2.2 -0.4]);
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-6 7]);
xlim([7,10]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
ylim([-6.5 6]);
xlim([7,10]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;



%% matrix completion d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MatrixCompletion/Record_MC_small_d40.mat')
% Record_CS_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MatrixCompletion\Record_MC_small_d40.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_MC_small_d40.mat')

niter = 100;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 1.5*nr*nc, 10));

Result = zeros(6,3,3,10);
p1 = permute(Record_MC_small_d40, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-1.3 -0.1]);
xlim([6.7 8]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
ylim([-4 4]);
xlim([6.7 8]);
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
ylim([-2 3.3]);
xlim([6.7 8]);
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;



%% matrix completion d = 80

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MatrixCompletion/Record_MC_small_d80.mat')
% Record_CS_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MatrixCompletion\Record_MC_small_d80.mat')
load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation8\Result\Record_MC_small_d80.mat')
niter = 100;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 1.5*nr*nc, 10));

Result = zeros(6,3,3,10);
p1 = permute(Record_MC_small_d80, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
xlim([7.3 9.5]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
ylim([-1.6, 0.3])
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
ylim([-5 5]);
xlim([7.3 9.5]);
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[2 3 4 6],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 13, 'Location', 'Northwest');
ylim([-2 3]);
xlim([7.3 9.5]);
xlabel('Log Sample Size', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 16, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;
