clear;clc;

%% initial parameters
nr   = 1e4;
nc   = 1e4;
r    = 10;
Nfac = 6;


%% data generation testing
dataOpts.nr = nr;
dataOpts.nc = nc;
dataOpts.r  = r;
dataOpts.Nfac = Nfac;
dataOpts.noise =  struct(...
    'type' , 'Gaussian',...
    'scale', 1.0,...
    'para' , 0.25);
dataOpts.sampling = 'wr';

[II, JJ, y, Theta_star, N] = DataGen_spMC(dataOpts);

%% Pivotal tuning
%% Pivotal tuning with power iteration
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

%% Pivotal tuning with lansvd
tuningOpts.nr      =  nr;
tuningOpts.nc      =  nc;
tuningOpts.N       =  N;
tuningOpts.B       =  200;
tuningOpts.alpha   =  0.20;
tuningOpts.mtd     =  'PROPACK';
tuningOpts.svdtol  =  1e-6;
tuningOpts.verbose =  1;
tuningOpts.loss    =  'L2';
tuningOpts.noise   =  dataOpts.noise;

lambdaHat  =  PivotalTuning_spMC(II, JJ, tuningOpts);


%% Pivotal tuning with svds
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


%% Solving
solvingOpts.verbose      =  0;
solvingOpts.mildverbose  =  1;
solvingOpts.nr           =  nr;
solvingOpts.nc           =  nc;
solvingOpts.N            =  N;
solvingOpts.lambda       =  lambdaHat;    
solvingOpts.eta          =  0.8;
solvingOpts.tol          =  1e-4;
solvingOpts.Lf           =  3e7;
solvingOpts.maxiter      =  1e3;
solvingOpts.matFormat    =  'factor';
Theta_init_dft.U = sparse(nr,1);
Theta_init_dft.S = 0;
Theta_init_dft.V = sparse(nc,1);
solvingOpts.Theta_init   =  Theta_init_dft;
solvingOpts.continuation_scaling = 1e-4;
solvingOpts.type         =  struct(...
    'name', 'L2', ...
    'para', 1.0);


[Theta_hat, svp] = MMAPG_sp(y, II, JJ, solvingOpts);


%% solve using MMAPG
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

noiseInd = 1;
Qtype.loss = 'L2';
Qtype.nr   = 100;
Qtype.nc   = 100;
Qtype.N    =  N;
Qtype.name = 'M';

[lambdaHat, ~] = PivotalTuning(Xw, 0.80, Qtype, noise(noiseInd));
if noiseInd == 1
    lambdaRun =  lambdaHat;
else
    lambdaRun =  0.05*lambdaHat;
end

tic;
[Theta_hat1, rank] = MMAPG(y, X, type, lambdaRun, 1e-4, 1e3, zeros(nr,nc));
elapsedTime = toc;


%% APGL

[Theta_hat] = APGL(nr,nc,'NNLS',@(Theta) Amap_sp(Theta, II, JJ, N, nr, nc), @(bb) ATmap_sp(bb, II, JJ, nr, nc), y, lambdaHat, 0);



Amap

