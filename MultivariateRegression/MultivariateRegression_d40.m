% Compressed Sensing
save_flag = true; % If doing trials please set this to 'false'.

niter = 40;
nr = 40;
nc = 40;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 2*nr*nc, 6));


tol = 1e-4; maxiter = 100;

Record = zeros(6, 3, niter, 3, 6); % method, record(loss, rank, time), niter, noise(G, C, L), 10 N

gauge = 1;
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

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1/64;

noise(3).type = 'Lognormal';
noise(3).para = 9;
noise(3).scale = 1/400;


Para_apgl.Clambda = [0.15, 10, 10]; % Gaussian, Cauchy, Lognormal

Para_l2.Clambda = [0.3, 10, 10];
%tuningPara_l2.Lf = [2e4, 2e4, 2e4];

Para_shrink.Clambda = [0.3, 0.1, 0.1];
Para_shrink.Ctau = [0.8, 0.3, 0.3];

Para_l1.Clambda = [0.3, 0.3, 0.4];
%tuningPara_l1.Lf = [3e4, 3e4, 3e4];

Para_wilcoxon.Clambda = [0.15, 0.2, 0.2];
wilcoxonLf = [5e3 5e4 5e5];
%tuningPara_wilcoxon.Lf = [1e4, 1e4, 1e4];

running = [1 0 1 1 0 1];

Para_huber.Clambda = [5, 0.1, 1.50];
huberPara = [2, 0.1, 1.5];
%tuningPara_huber.Lf = [2e4, 2e4, 2e4];

rng(10, 'twister');

%%
for noiseInd = 1 : 3
    for N_count = 1 : 6
        
        N = N_Cand(N_count);
        
        for trial = 1 : niter
            
            
            [X, y, Theta_star] = DataGen(nr, nc, N, r, noise(noiseInd), 'MR', 'small');
            
            %% APGL
            if running(1)
                type.name = 'APGL';
                Clambda = Para_apgl.Clambda(noiseInd);
                lambdaRun = Clambda * sqrt(N*(nr+nc));
                
                yVec = reshape(y', [], 1);
                
                tic;
                [Theta_hat1, ~, ~, sd, ~] = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X), @(bb) ATmap(bb, X), yVec, lambdaRun, 0);
                elapsedTime = toc;
                rank = length(find(sd>0));
                
                Record(1, 1, trial, noiseInd, N_count) = norm(Theta_hat1-Theta_star, 'fro')^2/(nr*nc);
                Record(1, 2, trial, noiseInd, N_count) = rank;
                Record(1, 3, trial, noiseInd, N_count) = elapsedTime;
                
                while (rank>r) && 0
                    Clambda = 2e3 * gauge ;
                    lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);
                    
                    tic;
                    [Theta_hat1, ~, ~, sd, ~] = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X), @(bb) ATmap(bb, X), y, lambdaRun, 0);
                    elapsedTime = toc;
                    rank = length(find(sd>0));
                    
                    Record(1, 1, trial, noiseInd, N_count) = norm(Theta_hat1-Theta_star, 'fro')^2/(nr*nc);
                    Record(1, 2, trial, noiseInd, N_count) = rank;
                    Record(1, 3, trial, noiseInd, N_count) = elapsedTime;
                    gauge = gauge + 1;
                end
                
                gauge = 1;
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['APGL L2 Error: ', num2str(norm(Theta_hat1-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat1-Theta_star, 'fro')))]);
                disp(' ');
            end
            
            %% L2 loss
            if running(2)
                type.name = 'L2';
                type.eta = 0.9;
                type.Lf = 5e2;
                Clambda = Para_l2.Clambda(noiseInd);
                lambdaRun = Clambda * sqrt(N*(nr+nc))/nr;
                
                tic;
                [Theta_hat2, rank] = MVAPG(y, X, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                elapsedTime = toc;
                
                Record(2, 1, trial, noiseInd, N_count) = norm(Theta_hat2-Theta_star, 'fro')^2/(nr*nc);
                Record(2, 2, trial, noiseInd, N_count) = rank;
                Record(2, 3, trial, noiseInd, N_count) = elapsedTime;
                
                while (rank>r) && 0
                    Clambda = 5e3 * gauge;
                    lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);
                    
                    tic;
                    [Theta_hat2, rank] = MVAPG(y, X, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                    elapsedTime = toc;
                    
                    Record(2, 1, trial, noiseInd, N_count) = norm(Theta_hat2-Theta_star, 'fro')^2/(nr*nc);
                    Record(2, 2, trial, noiseInd, N_count) = rank;
                    Record(2, 3, trial, noiseInd, N_count) = elapsedTime;
                    gauge = gauge + 1;
                end
                
                gauge = 1;
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['L2 Error: ', num2str(norm(Theta_hat2-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat2-Theta_star, 'fro')))]);
                disp(' ');
            end
            
%             if (norm(Theta_hat2-Theta_star, 'fro')^2/(nr*nc)>0.01)
%                 break;
%             end
            
            %% Robust Shrinkage
            
            if running(3)
                type.name = 'L2';
                type.eta = 0.8;
                type.Lf = 3e2;
                Clambda = Para_shrink.Clambda(noiseInd);
                lambdaRun = Clambda * sqrt(N*(nr+nc))/nr;
                Ctau = Para_shrink.Ctau(noiseInd);
                shrinkLevel = Ctau * sqrt(N/(nr+nc));
                
                ys = sign(y) .* min(abs(y), shrinkLevel);
                Xs = X;
                
                tic;
                [Theta_hat3, rank] = MVAPG(ys, Xs, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                elapsedTime = toc;
                
                Record(3, 1, trial, noiseInd, N_count) = norm(Theta_hat3-Theta_star, 'fro')^2/(nr*nc);
                Record(3, 2, trial, noiseInd, N_count) = rank;
                Record(3, 3, trial, noiseInd, N_count) = elapsedTime;
                
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['Robust L2 Error: ', num2str(norm(Theta_hat3-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat3-Theta_star, 'fro')))]);
                disp(' ');
            end
            
            %% L1 loss
            
            if running(4)
                type.name = 'L1';
                type.eta = 0.9;
                type.Lf = 3e5;
                Clambda = Para_l1.Clambda(noiseInd);
                lambdaRun = Clambda * sqrt(N*(nr+nc))/nr;
                
                tic;
                [Theta_hat4, rank] = MVAPG(y, X, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                elapsedTime = toc;
                
                Record(4, 1, trial, noiseInd, N_count) = norm(Theta_hat4-Theta_star, 'fro')^2/(nr*nc);
                Record(4, 2, trial, noiseInd, N_count) = rank;
                Record(4, 3, trial, noiseInd, N_count) = elapsedTime;
                
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['L1 Error: ', num2str(norm(Theta_hat4-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat4-Theta_star, 'fro')))]);
                disp(' ');
            end
            
            
            %% Huber loss
            
            if running(5)
                type.name = 'Huber'; type.para = huberPara(noiseInd);
                type.eta = 0.9;
                type.Lf = 5e1;
                Clambda = Para_huber.Clambda(noiseInd);
                lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);
                
                tic;
                [Theta_hat5, rank] = MVAPG(y, X, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                elapsedTime = toc;
                
                Record(5, 1, trial, noiseInd, N_count) = norm(Theta_hat5-Theta_star, 'fro')^2/(nr*nc);
                Record(5, 2, trial, noiseInd, N_count) = rank;
                Record(5, 3, trial, noiseInd, N_count) = elapsedTime;
                
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['Huber Error: ', num2str(norm(Theta_hat5-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat5-Theta_star, 'fro')))]);
                disp(' ');
            end
            
            %% Wilcoxon loss
            
            if running(6)
                type.name = 'Wilcoxon';
                type.eta = 0.9;
                type.Lf = wilcoxonLf(noiseInd);
                Clambda = Para_wilcoxon.Clambda(noiseInd);
                %lambdaRun = Clambda * sqrt(log(max(nr,nc))*max(nr,nc)/N);
                
                yw = y;
                Xw = X;
                
                alpha = 0.90;
                Qtype.name = 'V';
                Qtype.nr = nr;
                Qtype.nc = nc;
                Qtype.N = N;
                
                [lambdaHat, ~] = PivotalTuning(Xw, alpha, Qtype);
                lambdaRun = Clambda * lambdaHat;
                
                tic;
                [Theta_hat6, rank] = MVAPG(yw, Xw, type, lambdaRun, tol, maxiter, zeros(nr,nc));
                elapsedTime = toc;
                
                Record(6, 1, trial, noiseInd, N_count) = norm(Theta_hat6-Theta_star, 'fro')^2/(nr*nc);
                Record(6, 2, trial, noiseInd, N_count) = rank;
                Record(6, 3, trial, noiseInd, N_count) = elapsedTime;
                
                disp(' ');
                disp(['noiseInd: ', num2str(noiseInd), ' N_count: ', num2str(N_count)]);
                disp(['Wilcoxon Error: ', num2str(norm(Theta_hat6-Theta_star, 'fro')^2/(nr*nc)), ' Trial: ', num2str(trial)]);
                disp(['log error: ', num2str(log(norm(Theta_hat6-Theta_star, 'fro')))]);
                disp(' ');
            end
            
            %%
            
            % if sum(Error(trial, :)>0.05)>0
            %     break;
            % end
            
            
        end
    end
end
% options.tol   = 1e-6;
% options.issym = true;
% options.disp  = 0;
% options.v0    = randn(N,1);
% vvv = eigs(@(y)AAmap(AATmap(y, X, ones(N,1)), X, ones(N,1)),N,1,'LM',options);

if save_flag
    Record_MR_small_d40 = Record;
    save('Record_MR_small_d40.mat', 'Record_MR_small_d40');
end

%disp(num2str(log(Record(:,1,1,1,:)*1600)/2))
%disp(num2str(log(Record(:,1,:,2,5)*1600)/2))
%disp(num2str(log(Record(:,1,2,2,:)*1600)/2))
%disp(num2str(log(Record(:,1,3,2,:)*1600)/2))

% method, record(loss, rank, time), niter, noise(G, C, L), 10 N
%sv=svd(Theta_hat1);sv(sv>1e-4)
%sv=svd(Theta_hat2);sv(sv>1e-4)
%sv=svd(Theta_hat3);sv(sv>1e-4)
%sv=svd(Theta_hat4);sv(sv>1e-4)
%sv=svd(Theta_hat5);sv(sv>1e-4)
%sv=svd(Theta_hat6);sv(sv>1e-4)


