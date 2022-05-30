% Compressed Sensing
save_flag = false; % If doing trials please set this to 'false'.

niter = 3;
nr = 40;
nc = 40;
N = 1200;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(5*max(nr,nc)*r, 1.5*nr*nc, 10));
number_lambda = 20;
K = 5; % CV folds
eta = 0.95;
number_lambda = 20;
quan = 0.7:0.05:1;

tol = 1e-4; maxiter = 100;

Record = zeros(2, 4, niter, 3); % method, record(loss, rank, time), niter, noise(G, C, L), 10 N

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


running = [0 0 1 0 0 1];

rng(10, 'twister');

%%
for noiseInd = 1 : 3
    

        for trial = 1 : niter
            
            
            [X, y, Theta_star] = DataGen(nr, nc, N, r, noise(noiseInd), 'MC', 'small');

            
            %% Robust Shrinkage
            
            if running(3)
                type.name = 'L2';
                type.eta = 0.8;
                type.Lf = 3e5;
                Clambda_Cand = 0.2 * (1:number_lambda);
                fold = K;
                robust.flag = true;
                robust.para = eta;
                trim.flag = true;
                trim.para = quan;
                plot_flag = false;
                
                [ClambdaHat, ~, elapsedTime1, CtauHat, ~] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag);
                lambdaHat = ClambdaHat * sqrt((nr+nc)*N);
                tauHat = 2 * CtauHat * sqrt(N/(max(nr,nc)*log(max(nc,nr))));
                tic;
                ys = sign(y) .* min(abs(y), tauHat);
                Xs = X;
                [Theta_hat3, ~] = MMAPG(ys, Xs, type, lambdaHat, tol, maxiter, zeros(nr,nc));
                elapsedTime2 = toc;
                
                Record(1,1,trial, noiseInd) = norm(Theta_hat3-Theta_star, 'fro')^2;
                Record(1,2,trial, noiseInd) = elapsedTime1;
                Record(1,3,trial, noiseInd) = elapsedTime2;
                Record(1,4,trial, noiseInd) = elapsedTime1 + elapsedTime2;
            end

            
            %% Wilcoxon loss
            
            if running(6)
                type.name = 'Wilcoxon';
                type.eta = 0.9;
                type.Lf = 3e4;
                alpha = 0.95;
                Qtype.name = 'M';
                Qtype.nr = nr;
                Qtype.nc = nc;
                Qtype.N = N;
                % lambdaRun = Clambda * sqrt((nr+nc)*N);
                
                yw = y;
                Xw = X;
                for i = 1:N
                    radSign = (2 * (rand(1)>0.5) - 1);
                    yw(i) = y(i) * radSign;
                    Xw(:,:,i) = X(:,:,i) * radSign;
                end
                
                [lambdaHat, elapsedTime1] = PivotalTuning(Xw, alpha, Qtype);
                tic;
                [Theta_hat6, ~] = MMAPG(yw, Xw, type, 0.2*lambdaHat, tol, maxiter, zeros(nr,nc));
                elapsedTime2 = toc;
                
                Record(2,1,trial, noiseInd) = norm(Theta_hat6-Theta_star, 'fro')^2;
                Record(2,2,trial, noiseInd) = elapsedTime1;
                Record(2,3,trial, noiseInd) = elapsedTime2;
                Record(2,4,trial, noiseInd) = elapsedTime1 + elapsedTime2;
            end

            
        end
end
% options.tol   = 1e-6;
% options.issym = true;
% options.disp  = 0;
% options.v0    = randn(N,1);
% vvv = eigs(@(y)AAmap(AATmap(y, X, ones(N,1)), X, ones(N,1)),N,1,'LM',options);

if save_flag
    Record_MC_small_d40 = Record;
    save('Record_MC_small_d40.mat', 'Record_MC_small_d40');
end

% Record(1,:,1,1)
% Record(2,:,1,1)

(Record(1,:,1,1)+Record(1,:,2,1)+Record(1,:,3,1))/3
(Record(2,:,1,1)+Record(2,:,2,1)+Record(2,:,3,1))/3

(Record(1,:,1,2)+Record(1,:,2,2)+Record(1,:,3,2))/3
(Record(2,:,1,2)+Record(2,:,2,2)+Record(2,:,3,2))/3

(Record(1,:,1,3)+Record(1,:,2,3)+Record(1,:,3,3))/3
(Record(2,:,1,3)+Record(2,:,2,3)+Record(2,:,3,3))/3
