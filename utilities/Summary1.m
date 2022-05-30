
%% multivariate regression d = 40

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/MultivariateRegression/Record_MR_small_d40.mat')
% Record_MR_small_d40
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
% load('E:\mac\Documents\ResearchWork\robustWilcoxon\Simulation7\MultivariateRegression\Record_MR_small_d40.mat')
load('C:\Users\22055\Desktop\res\Record_MR_small_d40_ardesign0.mat')
niter = 40;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(500, 2000, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_MR_small_d40_ardesign0, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-1.1 0.1]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-5 7]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-8 5]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
load('C:\Users\22055\Desktop\res\Record_MR_small_d40_ardesign08.mat')
niter = 40;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(500, 2000, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_MR_small_d40_ardesign08, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([0.35 0.7]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-3 8]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
%ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-3 5]);
xlim([6 7.8]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;





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
    'FontSize', 18, 'Location', 'Northwest');
xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;



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
    'FontSize', 18, 'Location', 'Northwest');
% xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
    'FontSize', 18, 'Location', 'Northwest');
xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
    'FontSize', 18, 'Location', 'Northwest');
% xlim([1.5 5]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
load('C:\Users\22055\Desktop\res\Record_CS_small_d40.mat')
niter = 40;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(3200, 6400, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_CS_small_d40, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-1.6 -0.5]);
xlim([7.9 8.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-5 4]);
xlim([7.9 8.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2, 'LineWidth', 2.5, 'MarkerSize', 12); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
ylim([-6 3]);
xlim([7.9 8.9]);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;



%% compressed sensing d = 80

% load('/Users/leishi/Documents/ResearchWork/robustWilcoxon/Simulation7/CompressedSensing/Record_CS_small_d80.mat')
% Record_CS_small_d80
% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
load('C:\Users\22055\Desktop\res\Record_CS_small_d803.mat')
niter = 40;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(3200, 6400, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_CS_small_d803, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-1.2 0.3]); 
xlim([7.9,8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-4 6]) 
xlim([7.9,8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-5 5.5]); 
xlim([7.9,8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
load('C:\Users\22055\Desktop\res\Record_MC_small_d40.mat')

niter = 40;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(3200, 6400, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_MC_small_d40, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-1.6 -0.5]);
xlim([7.9 8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-5 4]);
xlim([7.9 8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
ylim([-2 2.5]);
xlim([7.9 8.9]);
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
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
load('C:\Users\22055\Desktop\res\Record_MC_small_d803.mat')
niter = 40;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(3200, 6400, 6));

Result = zeros(4,3,3,6);
p1 = permute(Record_MC_small_d803, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;
end

p2 = permute(Result, [4, 1, 3, 2]);

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],1,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
xlim([7.9 8.9]);
ylim([-1.2 -0.1])
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],2,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
xlim([7.9 8.9]);
ylim([-4 4])
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;

figure;
p = plot(log(N_Cand), log(p2(:,[1 2 3 4],3,1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
xlim([7.9 8.9]);
ylim([-2 3])
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 18, 'Location', 'Northwest');
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 17, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;
