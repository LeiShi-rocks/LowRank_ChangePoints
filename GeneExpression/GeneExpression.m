%% Data preprocessing
GeneChip = importdata('GeneChip.txt');
GeneChipUS = GeneChip.data; % Gene expression data from upstream pathways

GeneChipMore = importdata('GeneChipMore.txt');
GeneChipDSfull = GeneChipMore.data; % Gene expression data from downstream pathways

PathwayName = GeneChipMore.textdata(:, 1);
PathwayName = PathwayName(2: end); % remove the head

PLindex = find(strcmp(PathwayName, 'Plastoquinonebiosynthesis')); % Plastoquinone
CAindex = find(strcmp(PathwayName, 'Carotenoidbiosynthesis')); % Carotenoid
PHindex = find(strcmp(PathwayName, 'Phytosterolbiosynthesis')); % Phytosterol
CHindex = find(strcmp(PathwayName, 'PorphyrinChlorophyllmetabolism')); % Chlorophyll

position = [ones([1, size(PLindex)]), 2*ones([1, size(CAindex)]), ...
    3*ones([1, size(PHindex)]), 4*ones([1, size(CHindex)])];
PLposition = find(position == 1);
CAposition = find(position == 2);
PHposition = find(position == 3);
CHposition = find(position == 4);

GeneChipDS = GeneChipDSfull([PLindex; CAindex; PHindex; CHindex], :);

N = size(GeneChipDS, 2);
nr = size(GeneChipDS, 1);
nc = size(GeneChipUS, 1);

% centralization and standardization

meanGeneChipUS = mean(GeneChipUS, 2);
stdGeneChipUS = std(GeneChipUS, 0, 2);
GeneChipUS_st = (GeneChipUS - repmat(meanGeneChipUS, [1, 118]))./ repmat(stdGeneChipUS, [1, 118]);


meanGeneChipDS = mean(GeneChipDS, 2);
stdGeneChipDS = std(GeneChipDS, 0, 2);
GeneChipDS_st = (GeneChipDS - repmat(meanGeneChipDS, [1, 118]))./ repmat(stdGeneChipDS, [1, 118]);


%% Train/Test error comparison
perc = 0.8; % percentage of the train data
N_split = floor(N*perc);
N_train = N_split;
N_test = N - N_split;

% split the data
GeneChipDS_st_train = GeneChipDS_st(:, 1 : N_split);
GeneChipUS_st_train = GeneChipUS_st(:, 1 : N_split);
GeneChipDS_st_test = GeneChipDS_st(:, (N_split+1) : end);
GeneChipUS_st_test = GeneChipUS_st(:, (N_split+1) : end);

Record = zeros(4,3); % 4 methods, 3 indices(L2 and L1 test error, rank)


%% L2 Loss

tol = 1e-4; maxiter = 500;

type.name = 'L2';
type.eta = 0.9;
type.Lf = 1e4;
Clambda_Cand = 0.1 * (1:20);
fold = 5;
robust.flag = true;
robust.para = 0.9;
trim.flag = false;
plot_flag = false;

ClambdaHat =  MVAPG_CV(GeneChipDS_st_train, GeneChipUS_st_train, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);

% ClambdaHat = 1.0;
lambdaRun = ClambdaHat * sqrt(N_train*(nr+nc))/nr;
[Theta_hat1, rank1] = MVAPG(GeneChipDS_st_train, GeneChipUS_st_train, type, lambdaRun, tol, maxiter, zeros(nr,nc));

% Test error and rank
GeneChipDS_pre_test1 = Theta_hat1 * GeneChipUS_st_test;


Record(1, 1) = sum(sum((GeneChipDS_pre_test1 - GeneChipDS_st_test).^2)) / (N_test * nr);
Record(1, 2) = sum(sum(abs(GeneChipDS_pre_test1 - GeneChipDS_st_test))) / (N_test * nr);
Record(1, 3) = rank1;

%% Robust L2

tol = 1e-4; maxiter = 500;

type.name = 'L2';
type.eta = 0.9;
type.Lf = 5e2;
Clambda_Cand = 0.1 * (1:20);
fold = 5;
robust.flag = true;
robust.para = 0.9;
trim.flag = true;
trim.para = 0.7:0.05:1.0;
plot_flag = false;


[ClambdaHat, ~, ~, CtauHat, ~] =  MVAPG_CV(GeneChipDS_st, GeneChipUS_st, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);
shrinkLevel = (sqrt(N_train/(max(nr,nc)*log(max(nc,nr))))) * CtauHat;
% ClambdaHat = 1.0;

ys_train = sign(GeneChipDS_st_train) .* min(abs(GeneChipDS_st_train), shrinkLevel);
Xs_train = GeneChipUS_st_train;

lambdaRun = ClambdaHat * sqrt(N_train*(nr+nc))/nr;
[Theta_hat2, rank2] = MVAPG(ys_train, Xs_train, type, lambdaRun, tol, maxiter, zeros(nr,nc));

% Test error and rank
GeneChipDS_pre_test2 = Theta_hat2 * GeneChipUS_st_test;

Record(2, 1) = sum(sum((GeneChipDS_pre_test2 - GeneChipDS_st_test).^2)) / (N_test * nr);
Record(2, 2) = sum(sum(abs(GeneChipDS_pre_test2 - GeneChipDS_st_test))) / (N_test * nr);
Record(2, 3) = rank2;

%% L1 Loss

tol = 1e-4; maxiter = 500;

type.name = 'L1';
type.eta = 0.9;
type.Lf = 1e4;
Clambda_Cand = 0.1 * (1:20);
fold = 5;
robust.flag = true;
robust.para = 0.9;
trim.flag = false;
plot_flag = false;

ClambdaHat =  MVAPG_CV(GeneChipDS_st_train, GeneChipUS_st_train, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);

% ClambdaHat = 1.0;
lambdaRun = ClambdaHat * sqrt(N_train*(nr+nc))/nr;
[Theta_hat3, rank3] = MVAPG(GeneChipDS_st_train, GeneChipUS_st_train, type, lambdaRun, tol, maxiter, zeros(nr,nc));

% Test error and rank
GeneChipDS_pre_test3 = Theta_hat3 * GeneChipUS_st_test;

Record(3, 1) = sum(sum((GeneChipDS_pre_test3 - GeneChipDS_st_test).^2)) / (N_test * nr);
Record(3, 2) = sum(sum(abs(GeneChipDS_pre_test3 - GeneChipDS_st_test))) / (N_test * nr);
Record(3, 3) = rank3;


%% Wilcoxon Loss
tol = 1e-4; maxiter = 500;

type.name = 'Wilcoxon';
type.eta = 0.9;
type.Lf = 3e2;
Clambda = 0.2;
%lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);

yw_train = GeneChipDS_st_train;
Xw_train = GeneChipUS_st_train;

alpha = 0.90;
Qtype.name = 'V';
Qtype.nr = nr;
Qtype.nc = nc;
Qtype.N = N_train;

[lambdaHat, ~] = PivotalTuning(Xw_train, alpha, Qtype);
lambdaRun = Clambda * lambdaHat;
[Theta_hat4, rank4] = MVAPG(yw_train, Xw_train, type, lambdaRun, tol, maxiter, zeros(nr,nc));


% Test error and rank
GeneChipDS_pre_test4 = Theta_hat4 * GeneChipUS_st_test;

Record(4, 1) = sum(sum((GeneChipDS_pre_test4 - GeneChipDS_st_test).^2)) / (N_test * nr);
Record(4, 2) = sum(sum(abs(GeneChipDS_pre_test4 - GeneChipDS_st_test))) / (N_test * nr);
Record(4, 3) = rank4;




%% display the results of test error and the estimated rank
save('Record.mat', 'Record');
disp(' ');
disp([' Method', '    L2 error', '   ', 'L1 error', '  ', 'Estimated rank']);
disp(['   L2', '        ', num2str(Record(1,:), 4)]);
disp(['Robust L2', '    ', num2str(Record(2,:), 4)]);
disp(['   L1', '        ', num2str(Record(3,:), 4)]);
disp([' Rank ML', '     ', num2str(Record(4,:), 4)]);
disp(' ');


%% Factor analysis on full data

factor_plot = [0 0 0 1];

%% L2 Loss
if factor_plot(1)
    tol = 1e-4; maxiter = 500;
    
    type.name = 'L2';
    type.eta = 0.9;
    type.Lf = 1e4;
    Clambda_Cand = 0.1 * (1:20);
    fold = 5;
    robust.flag = true;
    robust.para = 0.9;
    trim.flag = false;
    plot_flag = true;
    
    ClambdaHat =  MVAPG_CV(GeneChipDS_st, GeneChipUS_st, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);
    
    % ClambdaHat = 1.0;
    lambdaRun = ClambdaHat * sqrt(N*(nr+nc))/nr;
    [Theta_hat1, rank1] = MVAPG(GeneChipDS_st, GeneChipUS_st, type, lambdaRun, tol, maxiter, zeros(nr,nc));
    
    % Factor analysis
    
    GeneChipDS_pre = Theta_hat1 * GeneChipUS_st;
    [U, S, ~] = svds(GeneChipDS_pre, rank1);
    
    loadings = U*S;
    
    colorSpec = [repmat([0 0 0], [size(PLindex), 1]); ...
        repmat([1 0 0], [size(CAindex), 1]); ...
        repmat([0 1 0], [size(PHindex), 1]); ...
        repmat([0 0 1], [size(CHindex), 1])];
    
    % first loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,1), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,1), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % second loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, -loadings(i,2), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, -loadings(:,2), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % third loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,3), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,3), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
end

%% Robust L2
if factor_plot(2)
    tol = 1e-4; maxiter = 500;
    
    type.name = 'L2';
    type.eta = 0.9;
    type.Lf = 1e4;
    Clambda_Cand = 0.1 * (1:20);
    fold = 5;
    robust.flag = true;
    robust.para = 0.9;
    trim.flag = true;
    trim.para = 0.7:0.05:1.0;
    plot_flag = true;
    
    
    [ClambdaHat, ~, ~, CtauHat, ~] =  MVAPG_CV(GeneChipDS_st, GeneChipUS_st, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);
    shrinkLevel = (sqrt(N/(max(nr,nc)*log(max(nc,nr))))) * CtauHat;
    % ClambdaHat = 1.0;
    
    ys = sign(GeneChipDS_st) .* min(abs(GeneChipDS_st), shrinkLevel);
    Xs = GeneChipUS_st;
    
    lambdaRun = ClambdaHat * sqrt(N*(nr+nc))/nr;
    [Theta_hat2, rank2] = MVAPG(ys, Xs, type, lambdaRun, tol, maxiter, zeros(nr,nc));
    
    % Factor analysis
    
    GeneChipDS_pre = Theta_hat2 * GeneChipUS_st;
    [U, S, ~] = svds(GeneChipDS_pre, rank2);
    
    loadings = U*S;
    
    colorSpec = [repmat([0 0 0], [size(PLindex), 1]); ...
        repmat([1 0 0], [size(CAindex), 1]); ...
        repmat([0 1 0], [size(PHindex), 1]); ...
        repmat([0 0 1], [size(CHindex), 1])];
    
    % first loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,1), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,1), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % second loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, -loadings(i,2), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, -loadings(:,2), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % third loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,3), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,3), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
end
%% L1 Loss
if factor_plot(3)
    tol = 1e-4; maxiter = 500;
    
    type.name = 'L1';
    type.eta = 0.9;
    type.Lf = 1e4;
    Clambda_Cand = 0.1 * (1:20);
    fold = 5;
    robust.flag = true;
    robust.para = 0.9;
    trim.flag = false;
    plot_flag = true;
    
    ClambdaHat =  MVAPG_CV(GeneChipDS_st, GeneChipUS_st, type, Clambda_Cand, fold, robust, trim, plot_flag, tol, maxiter);
    
    % ClambdaHat = 1.0;
    lambdaRun = ClambdaHat * sqrt(N*(nr+nc))/nr;
    [Theta_hat3, rank3] = MVAPG(GeneChipDS_st, GeneChipUS_st, type, lambdaRun, tol, maxiter, zeros(nr,nc));
    
    % Factor analysis
    
    GeneChipDS_pre = Theta_hat3 * GeneChipUS_st;
    [U, S, ~] = svds(GeneChipDS_pre, rank3);
    
    loadings = U*S;
    
    colorSpec = [repmat([0 0 0], [size(PLindex), 1]); ...
        repmat([1 0 0], [size(CAindex), 1]); ...
        repmat([0 1 0], [size(PHindex), 1]); ...
        repmat([0 0 1], [size(CHindex), 1])];
    
    % first loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,1), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,1), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % second loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, -loadings(i,2), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, -loadings(:,2), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
    
    % third loading
    figure;
    hold on;
    grid on;
    xlim([0, 63]);
    ylim([-10,10]);
    for i = 1 : nr
        plot(i, loadings(i,3), 'Marker', '.', 'MarkerSize', 15, 'Color', colorSpec(i,:));
        text(1:62, loadings(:,3), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    end
end

%% Wilcoxon Loss

if factor_plot(4)
    tol = 1e-4; maxiter = 500;
    
    type.name = 'Wilcoxon';
    type.eta = 0.9;
    type.Lf = 3e2;
    Clambda = 0.2;
    %lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);
    
    yw = GeneChipDS_st;
    Xw = GeneChipUS_st;
    
    alpha = 0.90;
    Qtype.name = 'V';
    Qtype.nr = nr;
    Qtype.nc = nc;
    Qtype.N = N;
    
    [lambdaHat, ~] = PivotalTuning(Xw, alpha, Qtype);
    lambdaRun = Clambda * lambdaHat;
    [Theta_hat4, rank4] = MVAPG(yw, Xw, type, lambdaRun, tol, maxiter, zeros(nr,nc));
    
    save('Theta_hat4.mat', 'Theta_hat4');
    % Factor analysis
    
    GeneChipDS_pre = Theta_hat4 * GeneChipUS_st;
    [U, S, ~] = svds(GeneChipDS_pre, rank4);
    
    loadings = U*S;
    
    colorSpec = [repmat([0 0 0], [size(PLindex), 1]); ...
        repmat([1 0 0], [size(CAindex), 1]); ...
        repmat([0 1 0], [size(PHindex), 1]); ...
        repmat([0 0 1], [size(CHindex), 1])];
    
    shapeSpec = [repmat('o', size(PLindex)), repmat('+', size(CAindex)),
        repmat('*', size(PHindex)), repmat('x', size(CHindex))];
    
    % first loading
    figure;
    hold on;
    % grid on;
    xlim([0, 65]);
    ylim([-12,12]);
    
    plot(PLposition, loadings(PLposition,1), 'o', 'MarkerSize', 7, 'Color', [0 0 0]);
    plot(CAposition, loadings(CAposition,1), '+', 'MarkerSize', 7, 'Color', [1 0 0]);
    plot(PHposition, loadings(PHposition,1), '*', 'MarkerSize', 7, 'Color', [0 1 0]);
    plot(CHposition, loadings(CHposition,1), 'x', 'MarkerSize', 7, 'Color', [0 0 1]);
    text((1:62)+0.4, loadings(:,1), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    legend({'Plastoquinone', 'Carotenoid', 'Phytosterol', 'Chlorophyll'}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Genes', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Factor coefficients', 'FontSize', 18, 'FontWeight', 'bold');
    ax = gca;
    set(ax, 'FontSize', 13, 'FontWeight', 'bold');
    plot(1:62, [ones(62,1)*S(1,1)/sqrt(nr), -ones(62,1)*S(1,1)/sqrt(nr)], 'LineStyle', '--', 'Color', 'k');
        
    
    % second loading
    figure;
    hold on;
    % grid on;
    xlim([0, 65]);
    ylim([-12,12]);
    
    plot(PLposition, -loadings(PLposition,2), 'o', 'MarkerSize', 7, 'Color', [0 0 0]);
    plot(CAposition, -loadings(CAposition,2), '+', 'MarkerSize', 7, 'Color', [1 0 0]);
    plot(PHposition, -loadings(PHposition,2), '*', 'MarkerSize', 7, 'Color', [0 1 0]);
    plot(CHposition, -loadings(CHposition,2), 'x', 'MarkerSize', 7, 'Color', [0 0 1]);
    text((1:62)+0.4, -loadings(:,2), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    legend({'Plastoquinone', 'Carotenoid', 'Phytosterol', 'Chlorophyll'}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Genes', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Factor coefficients', 'FontSize', 18, 'FontWeight', 'bold');
    ax = gca;
    set(ax, 'FontSize', 13, 'FontWeight', 'bold');
    plot(1:62, [ones(62,1)*S(2,2)/sqrt(nr), -ones(62,1)*S(2,2)/sqrt(nr)], 'LineStyle', '--', 'Color', 'k');
    
    % third loading
    figure;
    hold on;
    % grid on;
    xlim([0, 65]);
    ylim([-12,12]);
    
    plot(PLposition, loadings(PLposition,3), 'o', 'MarkerSize', 7, 'Color', [0 0 0]);
    plot(CAposition, loadings(CAposition,3), '+', 'MarkerSize', 7, 'Color', [1 0 0]);
    plot(PHposition, loadings(PHposition,3), '*', 'MarkerSize', 7, 'Color', [0 1 0]);
    plot(CHposition, loadings(CHposition,3), 'x', 'MarkerSize', 7, 'Color', [0 0 1]);
    text((1:62)+0.4, loadings(:,3), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    legend({'Plastoquinone', 'Carotenoid', 'Phytosterol', 'Chlorophyll'}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Genes', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Factor coefficients', 'FontSize', 18, 'FontWeight', 'bold');
    ax = gca;
    set(ax, 'FontSize', 13, 'FontWeight', 'bold');
    plot(1:62, [ones(62,1)*S(3,3)/sqrt(nr), -ones(62,1)*S(3,3)/sqrt(nr)], 'LineStyle', '--', 'Color', 'k');
    
    % fourth loading
    figure;
    hold on;
    % grid on;
    xlim([0, 65]);
    ylim([-10,10]);
    
    plot(PLposition, loadings(PLposition,4), 'o', 'MarkerSize', 7, 'Color', [0 0 0]);
    plot(CAposition, loadings(CAposition,4), '+', 'MarkerSize', 7, 'Color', [1 0 0]);
    plot(PHposition, loadings(PHposition,4), '*', 'MarkerSize', 7, 'Color', [0 1 0]);
    plot(CHposition, loadings(CHposition,4), 'x', 'MarkerSize', 7, 'Color', [0 0 1]);
    text((1:62)+0.4, loadings(:,4), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    legend({'Plastoquinone', 'Carotenoid', 'Phytosterol', 'Chlorophyll'}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Genes', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Factor coefficients', 'FontSize', 18, 'FontWeight', 'bold');
    ax = gca;
    set(ax, 'FontSize', 13, 'FontWeight', 'bold');
    plot(1:62, [ones(62,1)*S(4,4)/sqrt(nr), -ones(62,1)*S(4,4)/sqrt(nr)], 'LineStyle', '--', 'Color', 'k');
    
    % fifth loading
    figure;
    hold on;
    % grid on;
    xlim([0, 65]);
    ylim([-10,10]);
    
    plot(PLposition, loadings(PLposition,5), 'o', 'MarkerSize', 7, 'Color', [0 0 0]);
    plot(CAposition, loadings(CAposition,5), '+', 'MarkerSize', 7, 'Color', [1 0 0]);
    plot(PHposition, loadings(PHposition,5), '*', 'MarkerSize', 7, 'Color', [0 1 0]);
    plot(CHposition, loadings(CHposition,5), 'x', 'MarkerSize', 7, 'Color', [0 0 1]);
    text((1:62)+0.4, loadings(:,5), arrayfun(@num2str, 1:62, 'UniformOutput', false));
    legend({'Plastoquinone', 'Carotenoid', 'Phytosterol', 'Chlorophyll'}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Genes', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Factor coefficients', 'FontSize', 18, 'FontWeight', 'bold');
    ax = gca;
    set(ax, 'FontSize', 13, 'FontWeight', 'bold');
    plot(1:62, [ones(62,1)*S(5,5)/sqrt(nr), -ones(62,1)*S(5,5)/sqrt(nr)], 'LineStyle', '--', 'Color', 'k');
end




