function [y, X, outInfo] = DataGen_ChangePoints(opts)

%% parameters

nr = setOpts(opts, 'nr', 50);
nc = setOpts(opts, 'nc', 50);
N  = setOpts(opts, 'N',  2000);
r  = setOpts(opts, 'r',  5);
s  = setOpts(opts, 's',  1); % number of segments = number of change points + 1

noise_dft  =  struct(...
    'type',  'Gaussian',...
    'scale', 1.0,...
    'para',  1.0);
design_dft =  struct(...
    'type',  'CS',...
    'dist',  'Gaussian',...
    'scale', 1.0,...
    'para',  1.0);

noise  = setOpts(opts, 'noise', noise_dft);
design = setOpts(opts, 'design', design_dft);
mech   = setOpts(opts, 'mech' , 'random');
Theta_star = setOpts(opts, 'Theta_star', zeros(nr, nc, s));

Nsum   = sum(N);

%% generating true signals

if strcmp(mech, 'random')
    for k = 1:s
        [U,~,~]     =  svd(normrnd(0,1,[nr, 100]), 'econ');
        [V,~,~]     =  svd(normrnd(0,1,[nc, 100]), 'econ');
        Theta_star(:,:,k)  =  U(:, 1:r) * V(:, 1:r)' / sqrt(r);
    end
end
        

%% generating data

switch design.type
    case 'CS'
        X = design.scale * normrnd(0, design.para, [nr, nc, Nsum]);
        epsilon = noise.scale * normrnd(0, noise.para, [Nsum, 1]);
        y = zeros(Nsum, 1);
        Nstart = 1;
        Nend   = 0;
        for k = 1:s
            Nend = N(k) + Nend;
            ymed = zeros(N(k), 1);
            for i = 1:N(k)
                ymed(i) = sum(sum(X(:,:,i) .* Theta_star(:,:,k))) + epsilon(i);
            end
            y(Nstart: Nend) = ymed;
            Nstart = Nend + 1;
        end
        
    case 'MR'
        X = design.scale * normrnd(0, design.para, [N, nr]);
        epsilon = noise.scale * normrnd(0, noise.para, [N, nr]);
        y = X * Theta_star + epsilon;
        
    otherwise
        error('Not defined!');
end

outInfo.Theta = Theta_star;













