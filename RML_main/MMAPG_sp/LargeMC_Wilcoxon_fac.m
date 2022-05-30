if 1
    clear;clc;
end
%% initial parameters
% nr   = 1e4;
% nc   = 1e4;
% r    = 10;
% Nfac = 6;
if 1
    nr   = 5e4;
    nc   = 5e4;
    r    = 10;
    Nfac = 6;
end

%% data generation testing
if 1
    dataOpts.nr = nr;
    dataOpts.nc = nc;
    dataOpts.r  = r;
    dataOpts.Nfac = Nfac;
%     dataOpts.noise =  struct(...
%         'type' , 'Gaussian',...
%         'scale', 1.0,...
%         'para' , 0.25);
    dataOpts.noise = struct(...
         'type' , 'Cauchy',...
         'scale', 1/64,...
         'para' , 1);
    dataOpts.sampling = 'wr';
    dataOpts.matFormat = 'factor';
    
    [II, JJ, y, Theta_star, N] = DataGen_spMC(dataOpts);
end

%% Pivotal tuning
%% Pivotal tuning with power iteration
if 0
    tuningOpts.nr      =  nr;
    tuningOpts.nc      =  nc;
    tuningOpts.N       =  N;
    tuningOpts.B       =  100;
    tuningOpts.alpha   =  0.20;
    tuningOpts.mtd     =  'PowerIteration';
    tuningOpts.svdtol  =  1e-6;
    tuningOpts.verbose =  1;
    tuningOpts.loss    =  'L2';
    tuningOpts.noise   =  dataOpts.noise;
    
    lambdaHat  =  PivotalTuning_spMC(II, JJ, tuningOpts);
end

%% Pivotal tuning with lansvd
if 1
    tuningOpts.nr      =  nr;
    tuningOpts.nc      =  nc;
    tuningOpts.N       =  N;
    tuningOpts.B       =  200;
    tuningOpts.alpha   =  0.20;
    tuningOpts.mtd     =  'PROPACK';
    tuningOpts.svdtol  =  1e-6;
    tuningOpts.verbose =  1;
    tuningOpts.loss    =  'Wilcoxon';
    tuningOpts.noise   =  dataOpts.noise;
    
    lambdaHat  =  PivotalTuning_spMC(II, JJ, tuningOpts);
end

%% Pivotal tuning with svds
if 0
    tuningOpts.nr      =  nr;
    tuningOpts.nc      =  nc;
    tuningOpts.N       =  N;
    tuningOpts.B       =  100;
    tuningOpts.alpha   =  0.20;
    tuningOpts.mtd     =  'svds';
    tuningOpts.svdtol  =  1e-6;
    tuningOpts.verbose =  1;
    tuningOpts.loss    =  'L2';
    tuningOpts.noise   =  dataOpts.noise;
    
    lambdaHat  =  PivotalTuning_spMC(II, JJ, tuningOpts);
end

%% Solving
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


[Theta_hat, svp] = MMAPG_sp(y, II, JJ, solvingOpts);


%% solve using MMAPG
if 0
    X = zeros(nr, nc, N);
    for i = 1:N
        X(II(i),JJ(i),i) = sqrt(nr*nc);
    end
    
    noise(1).type = 'Gaussian';
    noise(1).para = 0.25;
    noise(1).scale = 1;
    
    noise(2).type = 'Cauchy';
    noise(2).para = 1;
    noise(2).scale = 1/64;
    
    noise(3).type = 'Lognormal';
    noise(3).para = 9;
    noise(3).scale = 1/400;
    
    %[X, y, Theta_star] = DataGen(nr, nc, N, r, noise(1), 'MC', 'small');
    
    type.name = 'L2';
    type.eta = 0.8;
    type.Lf = 3e4;
    %                 Clambda = Para_l2.Clambda(noiseInd);
    %                 lambdaRun = Clambda * sqrt(2*N*log(nr+nc)/nc);
    
    yw = y;
    Xw = X;
    
    noiseInd   =  1;
    Qtype.loss =  'L2';
    Qtype.nr   =  100;
    Qtype.nc   =  100;
    Qtype.N    =  N;
    Qtype.name =  'M';
    
    [lambdaHat, ~] = PivotalTuning(Xw, 0.80, Qtype, noise(noiseInd));
    if noiseInd == 1
        lambdaRun =  lambdaHat;
    else
        lambdaRun =  0.05*lambdaHat;
    end
    
    tic;
    [Theta_hat1, rank] = MMAPG(y, X, type, lambdaRun, 1e-4, 1e3, zeros(nr,nc));
    elapsedTime = toc;
end





