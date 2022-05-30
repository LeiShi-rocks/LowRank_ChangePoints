% Compressed Sensing
clear; clc;
save_flag = true; % If doing trials please set this to 'false'.

niter   =   100;
nr      =   80;
nc      =   80;
r       =   5;
N_Cand  =   ceil(linspace(5*max(nr,nc)*r, 2*nr*nc, 6));

tol     =   1e-4;
maxiter =   200;

Record  =   zeros(3, 4, niter, 3, 6);
% method, record(loss, rank, time, TotalIteration), niter, noise(G, C, L), 6 N

gauge   =   1;

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1/64;

noise(3).type = 'Lognormal';
noise(3).para = 9;
noise(3).scale = 1/400;

% Para_wilcoxon.Clambda = [0.15, 0.2, 0.2];
wilcoxonLf = [5e3 5e4 8e5];
% tuningPara_wilcoxon.Lf = [1e4, 1e4, 1e4];

running = [1 1 1];

rng(11, 'twister');

%%
for noiseInd = 1 : 3

    for N_count = 3 : 3
        
        N = N_Cand(N_count);
        
        for trial = 1 : niter
            
            disp('-------------------------------------------------------');
            disp(['== Running noise ', num2str(noiseInd), ' ==== ', ...
                ' N_count ', num2str(N_count), ' ===== ', ...
                ' trial ', num2str(trial), ' =====']);
            disp('-------------------------------------------------------');
            
            [X, y, Theta_star] = DataGen(nr, nc, N, r, noise(noiseInd), 'MR', 'small');
            
            alpha      =  0.20;
            Qtype.name =  'V';
            Qtype.nr   =  nr;
            Qtype.nc   =  nc;
            Qtype.N    =  N;
            Qtype.loss =  'wilcoxon';
            % Clambda = Para_wilcoxon.Clambda(noiseInd);
            
            [lambdaHat, ~] = PivotalTuning(X, alpha, Qtype);
            % lambda = Clambda * lambdaHat;
            lambda = lambdaHat;
            
            %% Wilcoxon loss with first order solver
            
            if running(1)
                % initialization of opts parameters
                opts.type     = struct(...
                    'name', 'Wilcoxon', ...
                    'para', 1.0);
                
                opts.Hessian  = struct(...
                    'name'  , 'const_diag', ...
                    'para'  , 1, ...
                    'Lf'    , wilcoxonLf(noiseInd));
                
                opts.t_search = struct(...
                    'name' , 'line_search',...
                    'para' , 0.4);
                
                opts.lambda    =  lambda;
                opts.eta       =  0.8;
                opts.verbose   =  0;
                opts.mildverbose  =  0;

                tic;
                [Theta_hat1, rank, outInfo] = MVPQN(y, X, opts);
                elapsedTime = toc;
                
                Record(1, 1, trial, noiseInd, N_count) = sum(sum( (Theta_hat1-Theta_star).^2 ))/(nr*nc);
                Record(1, 2, trial, noiseInd, N_count) = rank;
                Record(1, 3, trial, noiseInd, N_count) = elapsedTime;
                Record(1, 4, trial, noiseInd, N_count) = outInfo.TotalIteration;
                
                disp(' ');
                disp(['Method: ', 'Proximal']);
                disp(['Wilcoxon Error: ', num2str(sum(sum( (Theta_hat1-Theta_star).^2 ))/(nr*nc)), ...
                    ' |log error: ', num2str(log(norm(Theta_hat1-Theta_star, 'fro'))), ...
                    ' |rank: ', num2str(rank)]);
                disp(['Elapsed time: ', num2str(elapsedTime), ...
                    ' |Total Iteration: ', num2str(outInfo.TotalIteration)]);
                disp(' ');
            end
            
            %% Wilcoxon loss with second order solver(LSR1, 1 step)

            if running(2)
                % initialization of opts parameters
                opts.type     = struct(...
                    'name', 'Wilcoxon', ...
                    'para', 1.0);
                
                opts.Hessian  = struct(...
                    'name'  , 'LSR1', ...
                    'para'  , 1, ...
                    'Lf'    , wilcoxonLf(noiseInd));
                
                opts.t_search = struct(...
                    'name' , 'line_search',...
                    'para' , 0.4);
                
                opts.lambda    =  lambda;
                opts.eta       =  0.8;
                opts.verbose   =  0;
                opts.mildverbose  =  0;

                tic;
                [Theta_hat2, rank, outInfo] = MVPQN(y, X, opts);
                elapsedTime = toc;
                
                Record(2, 1, trial, noiseInd, N_count) = sum(sum((Theta_hat2-Theta_star).^2))/(nr*nc);
                Record(2, 2, trial, noiseInd, N_count) = rank;
                Record(2, 3, trial, noiseInd, N_count) = elapsedTime;
                Record(2, 4, trial, noiseInd, N_count) = outInfo.TotalIteration;
                
                disp(' ');
                disp(['Method: ', 'LSR1']);
                disp(['Wilcoxon Error: ', num2str(sum(sum( (Theta_hat2-Theta_star).^2 ))/(nr*nc)), ...
                    ' |log error: ', num2str( log( norm( Theta_hat2-Theta_star , 'fro') ) ),...
                    ' |rank: ', num2str(rank)]);
                disp(['Elapsed time: ', num2str(elapsedTime),...
                    ' |Total Iteration: ', num2str(outInfo.TotalIteration)]);
                disp(' ');
            end
            
            %% Wilcoxon loss with second order solver(LBFGS, 1 step)
            
            if running(3)
                % initialization of opts parameters
                opts.type     = struct(...
                    'name' , 'Wilcoxon', ...
                    'para' , 1.0);
                
                opts.Hessian  = struct(...
                    'name' , 'LBFGS', ...
                    'para' , 1, ...
                    'Lf'   , wilcoxonLf(noiseInd));
                
                opts.t_search = struct(...
                    'name' , 'line_search',...
                    'para' , 0.4);
                
                opts.lambda    =  lambda;
                opts.eta       =  0.8;
                opts.verbose   =  0;
                opts.mildverbose  =  0;

                tic;
                [Theta_hat3, rank, outInfo] = MVPQN(y, X, opts);
                elapsedTime = toc;
                
                Record(3, 1, trial, noiseInd, N_count) = sum(sum((Theta_hat3-Theta_star).^2))/(nr*nc);
                Record(3, 2, trial, noiseInd, N_count) = rank;
                Record(3, 3, trial, noiseInd, N_count) = elapsedTime;
                Record(3, 4, trial, noiseInd, N_count) = outInfo.TotalIteration;
                
                disp(' ');
                disp(['Method: ', 'LBFGS']);
                disp(['Wilcoxon Error: ', num2str(sum(sum( (Theta_hat3-Theta_star).^2 ))/(nr*nc)), ...
                    ' |log error: ', num2str(log(norm(Theta_hat3-Theta_star, 'fro'))), ...
                    ' |rank: ', num2str(rank)]);
                disp(['Elapsed time: ', num2str(elapsedTime),...
                    ' |Total Iteration: ', num2str(outInfo.TotalIteration)]);
                disp(' ');
            end        
        end
    end
end
% options.tol   = 1e-6;
% options.issym = true;
% options.disp  = 0;
% options.v0    = randn(N,1);
% vvv = eigs(@(y)AAmap(AATmap(y, X, ones(N,1)), X, ones(N,1)),N,1,'LM',options);

if save_flag
    Record_MR_small_d80 = Record;
    save('Record_MR_small_d80.mat', 'Record_MR_small_d80');
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


