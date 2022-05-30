% Compressed Sensing Tuning

save_flag = true;

niter = 50;
nr = 40;
nc = 40;
N = 1000;
r = 5;
X = zeros(nr, nc, N);
y = zeros(N, 1);
K = 5; % CV folds
eta = 0.95;
tol = 1e-4;
maxiter = 100;

number_lambda = 20;
quan = 0.7:0.05:1;

%% small signal setting (Fan2016)

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1/64;

noise(3).type = 'Lognormal';
noise(3).para = 9;
noise(3).scale = 1/400;

noise(4).type = 'Chi2';
noise(4).para = 3;
noise(4).scale = 0.1;

% large signal setting (Negahban2011)

% noise(1).type = 'Gaussian';
% noise(1).para = 1;
% noise(1).scale = 1;
%
% noise(2).type = 'Cauchy';
% noise(2).para = 1;
% noise(2).scale = 0.1;
%
% noise(3).type = 'Lognormal';
% noise(3).para = 1;
% noise(3).scale = 1;

running = [0 0 0 0 0 0 1];

% choose lambda according to C*sqrt(nr*r*log(nr) / N) = 0.8589
Clambda_CS = zeros(7, 6, 3, niter); % seven methods(APGL-L2, L2, Robust-L2, L1, Huber, Wilcoxon-cv, Wilcoxon-pivotal),
% 5 indices(CVlambda, CVtau for Fan2016, tuning time, solving time, total time, loss), 3 noise, niter;
cvRec = zeros(K, 1);
cvScore = zeros(number_lambda,1);

rng(11, 'twister');

for noiseInd = 1 : 3
    for trial = 1 : niter
        
        [X, y, Theta_star] = DataGen(nr, nc, N, r, noise(noiseInd), 'CS', 'small');
        
        %% APGL
        if running(1)
            type.name = 'APGL';
            Clambda_Cand = 0.2 * (1:number_lambda);
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = false;
            plot_flag = true;
            
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tic;
            Theta_hat1 = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X), @(bb) ATmap(bb, X), y, lambdaHat, 0);
            elapsedTime2 = toc;
            
            Clambda_CS(1, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(1, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(1, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(1, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(1, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(1, 6, noiseInd, trial) = norm(Theta_hat1-Theta_star, 'fro')^2/(nr*nc) ;
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['APGL L2 Error: ', num2str(Clambda_CS(1, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% L2 loss
        if running(2)
            type.name = 'L2';
            type.eta = 0.8;
            type.Lf = 3e4;
            Clambda_Cand = 0.2 * (1:number_lambda);
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = false;
            plot_flag = false;
            
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tic;
            [Theta_hat2, ~] = MMAPG(y, X, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            
            Clambda_CS(2, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(2, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(2, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(2, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(2, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(2, 6, noiseInd, trial) = norm(Theta_hat2-Theta_star, 'fro')^2/(nr*nc) ;
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['L2 Error: ', num2str(Clambda_CS(2, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% Robust-L2 (Fan2016)
        if running(3)
            type.name = 'L2';
            type.eta = 0.8;
            type.Lf = 3e4;
            Clambda_Cand = 0.2 * (1:number_lambda);
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = true;
            trim.para = quan;
            plot_flag = false;
            
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tauHat = 2 * CtauHat * sqrt(N/(nr+nc));
            tic;
            ys = sign(y) .* min(abs(y), tauHat);
            Xs = X;
            [Theta_hat3, ~] = MMAPG(ys, Xs, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            
            Clambda_CS(3, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(3, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(3, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(3, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(3, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(3, 6, noiseInd, trial) = norm(Theta_hat3-Theta_star, 'fro')^2/(nr*nc);
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['Robust L2 Error: ', num2str(Clambda_CS(3, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% L1 loss
        if running(4)
            type.name = 'L1';
            type.eta = 0.9;
            type.Lf = 1e5;
            Clambda_Cand = 0.2 * (1:number_lambda);
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = false;
            plot_flag = false;
            
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tic;
            [Theta_hat4, ~] = MMAPG(y, X, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            Clambda_CS(4, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(4, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(4, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(4, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(4, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(4, 6, noiseInd, trial) = norm(Theta_hat4-Theta_star, 'fro')^2/(nr*nc) ;
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['L1 Error: ', num2str(Clambda_CS(4, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% Huber loss
        
        if running(5)
            type.name = 'Huber'; type.para = 1.5;  % choosen as Sara 2017
            type.eta = 0.8; type.Lf = 1e5;
            Clambda_Cand = 0.2 * (1:number_lambda);
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = false;
            plot_flag = false;
            
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tic;
            [Theta_hat5, ~] = MMAPG(y, X, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            
            Clambda_CS(5, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(5, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(5, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(5, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(5, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(5, 6, noiseInd, trial) = norm(Theta_hat5-Theta_star, 'fro')^2/(nr*nc) ;
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['Huber Error: ', num2str(Clambda_CS(5, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% Wilcoxon-CV
        
        if running(6)
            type.name = 'Wilcoxon';
            type.eta = 0.9;
            type.Lf = 3e4;
            Clambda_Cand = 5 * (1:number_lambda) + 30;
            fold = K;
            robust.flag = true;
            robust.para = eta;
            trim.flag = false;
            plot_flag = 0;
            
            yw = y;
            Xw = X;
            [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(yw, Xw, type, Clambda_Cand, fold, robust, trim, plot_flag);
            lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
            tic;
            [Theta_hat6, ~] = MMAPG(yw, Xw, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            
            Clambda_CS(6, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(6, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(6, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(6, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(6, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(6, 6, noiseInd, trial) = norm(Theta_hat6-Theta_star, 'fro')^2/(nr*nc);
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['Wilcoxon-cv Error: ', num2str(Clambda_CS(6, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
        %% Wilcoxon-pivotal
        if running(7)
            type.name = 'Wilcoxon';
            type.eta = 0.9;
            type.Lf = 3e4;
            alpha = 0.95;
            Qtype.name = 'M';
            Qtype.nr = nr;
            Qtype.nc = nc;
            Qtype.N = N;
            
            yw = y;
            Xw = X;
            [lambdaHat, elapsedTime1] = PivotalTuning(Xw, alpha, Qtype);
            tic;
            [Theta_hat7, ~] = MMAPG(yw, Xw, type, 0.2*lambdaHat, tol, maxiter, zeros(nr,nc));
            elapsedTime2 = toc;
            ClambdaHat = 0.2*lambdaHat/(sqrt((nr+nc)*N));
            CtauHat = Inf;
            
            Clambda_CS(7, 1, noiseInd, trial) = ClambdaHat;
            Clambda_CS(7, 2, noiseInd, trial) = CtauHat;
            Clambda_CS(7, 3, noiseInd, trial) = elapsedTime1;
            Clambda_CS(7, 4, noiseInd, trial) = elapsedTime2;
            Clambda_CS(7, 5, noiseInd, trial) = elapsedTime1 + elapsedTime2;
            Clambda_CS(7, 6, noiseInd, trial) = norm(Theta_hat7-Theta_star, 'fro')^2/(nr*nc);
            
            disp(' ');
            disp(['noiseInd: ', num2str(noiseInd)]);
            disp(['Wilcoxon-pivotal Error: ', num2str(Clambda_CS(7, 6, noiseInd, trial)), ' noiseInd: ', num2str(noiseInd), ' Trial: ', num2str(trial)]);
            disp(' ');
        end
        
    end
end

if save_flag
    Record_CS_small_d40_tuning = Clambda_CS;
    save('Record_CS_small_d40_tuning.mat', 'Record_CS_small_d40_tuning');
end
