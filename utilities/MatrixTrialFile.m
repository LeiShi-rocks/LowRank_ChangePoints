
% Compressed Sensing tuning (large signal)

niter = 1;
nr = 40;
nc = 40; 
N = 1000;
r = 5;
X = zeros(nr, nc, N);
y = zeros(N, 1);
K = 5; % CV folds
tol = 1e-4;
maxiter = 100;

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy'; 
noise(2).para = 1;
noise(2).scale = 1/64;

noise(3).type = 'Lognormal';
noise(3).para = 9;
noise(3).scale = 1/1000;

[X, y, Theta_star] = DataGen(nr, nc, N, r, noise(1), 'MC', 'small');

%% APGL
tic;
[Theta_hat, ~, ~, sd, ~] = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X), @(bb) ATmap(bb, X), y, 1, 0);
elapsedTime = toc;

norm(Theta_hat-Theta_star, 'fro')^2/(nr*nc)





%% Wilcoxon loss

type.name = 'Wilcoxon';
type.eta = 0.9;
type.Lf = 1e4;

yw = y;
Xw = X;

tic;
[Theta_hat3, rank] = MMAPG(yw, Xw, type, 2, tol, maxiter, zeros(nr,nc));
elapsedTime = toc;

norm(Theta_hat3-Theta_star, 'fro')^2/(nr*nc)



%% Tuning
type.name = 'L2';
type.eta = 0.8;
type.Lf = 1e5;
Clambda_Cand = 0.2 * (16:20);
fold = K;
robust.flag = true; robust.para = 0.95;
% robust.flag = false;
trim.flag = true; trim.para = 0.9:0.02:1;
plot_flag = true;

[ClambdaHat, oneSDScore, elapsedTime, CtauHat] = APG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);







