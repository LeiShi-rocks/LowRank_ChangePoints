% Simulation Study for large matrix completion
clear;clc;

dim   =  [5e3, 1e4, 5e4];
r     =  10;
Nfac  =  6;
niter =  25;

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1/64;

saving_flag = 0;
Record  =  struct(...
    'error', zeros(length(dim), length(noise), niter),...
    'rank' , zeros(length(dim), length(noise), niter),...
    'timeTuning' , zeros(length(dim), length(noise), niter),...
    'timeSolving' , zeros(length(dim), length(noise), niter),...
    'numIter', zeros(length(dim), length(noise), niter)...
    );

for dimInd = 3:3
    for noiseInd = 2:2
        for iter = 1:niter
            disp('-----------------------------------------');
            disp(['Runing: dim = ', num2str(dimInd),...
                ' noise = ', num2str(noiseInd),...
                ' iter = ', num2str(iter)]);
            % data generating
            nr = dim(dimInd); nc = dim(dimInd);
            dataOpts.nr = nr;
            dataOpts.nc = nc;
            dataOpts.r  = r;
            dataOpts.Nfac = Nfac;
            dataOpts.noise = noise(noiseInd);
            dataOpts.sampling = 'wr';
            dataOpts.matFormat = 'factor';
            
            [II, JJ, y, Theta_star, N] = DataGen_spMC(dataOpts);
            
            % Pivotal tuning
            tuningOpts.nr      =  nr;
            tuningOpts.nc      =  nc;
            tuningOpts.N       =  N;
            tuningOpts.B       =  100;
            tuningOpts.alpha   =  0.20;
            tuningOpts.mtd     =  'PROPACK';
            tuningOpts.svdtol  =  1e-6;
            tuningOpts.verbose =  0;
            tuningOpts.loss    =  'Wilcoxon';
            tuningOpts.noise   =  dataOpts.noise;
            
            [lambdaHat, timeTuning]  =  PivotalTuning_spMC(II, JJ, tuningOpts);
            
            
            % Solving
            solvingOpts.verbose      =  0;
            solvingOpts.mildverbose  =  1;
            solvingOpts.nr           =  nr;
            solvingOpts.nc           =  nc;
            solvingOpts.N            =  N;
            solvingOpts.lambda       =  lambdaHat;
            solvingOpts.eta          =  0.7;
            solvingOpts.tol          =  1e-4;
            solvingOpts.Lf           =  3e8; % 1e4 -> 3e7;
            solvingOpts.maxiter      =  1e3;
            solvingOpts.matFormat    =  'factor';
            Theta_init_dft.U = sparse(nr,1);
            Theta_init_dft.S = 0;
            Theta_init_dft.V = sparse(nc,1);
            solvingOpts.Theta_init   =  Theta_init_dft;
            solvingOpts.continuation_scaling = 1e-4;
            solvingOpts.type         =  struct(...
                'name', 'Wilcoxon', ...
                'para', 1.0);
            
           [Theta_hat, svp, outInfo] =  MMAPG_sp(y, II, JJ, solvingOpts);
           
           % Collecting results
           square_error = compute_SE(Theta_hat, Theta_star, solvingOpts.matFormat);
           Record.error(dimInd, noiseInd, iter)          =   square_error;
           Record.rank(dimInd, noiseInd, iter)           =   svp;
           Record.timeTuning(dimInd, noiseInd, iter)     =   timeTuning;
           Record.timeSolving(dimInd, noiseInd, iter)    =   outInfo.elapsedTime;
           Record.numIter(dimInd, noiseInd, iter)        =   outInfo.numIter;
           
           disp(['Result: error = ', num2str(square_error), ' rank = ', num2str(svp)]);
           disp(['Tuning = ', num2str(timeTuning),...
               ' Solving = ', num2str(outInfo.elapsedTime),...
               ' numIter = ', num2str(outInfo.numIter)]);
           disp(' ')
        end
    end
end

if save_flag
    Record_largeMC = Record;
    save('Record_largeMC.mat', 'Record_largeMC');
end



