function [II, JJ, y, Theta_star, N] = DataGen_spMC(opts)

% opts should contain:
% nr, nc, Nfrac, r, noise, sampling

nr = setOpts(opts, 'nr', 100);
nc = setOpts(opts, 'nc', 100);
Nfac  = setOpts(opts, 'Nfac' , 5);
r  = setOpts(opts, 'r' , 5);
matFormat = setOpts(opts, 'matFormat', 'standard');

noise_dft =  struct(...
    'type' , 'Gaussian',...
    'scale', 1.0,...
    'para' , 0.25);
noise     =  setOpts(opts, 'noise', noise_dft);

sampling  =  setOpts(opts, 'sampling', 'wr');   % with/without replacement
dr        =  r*(nr+nc-r);
N         =  ceil(Nfac * dr);
II        =  zeros(N, 1);
JJ        =  zeros(N, 1);
y         =  zeros(N, 1);
cnt       =  0;


% Generate a true matrix
[Ufull,~,~] = svd(normrnd(0,1,[nr, 100]), 'econ');
[Vfull,~,~] = svd(normrnd(0,1,[nc, 100]), 'econ');
% Theta_star = U * V'/sqrt(r);
U  = Ufull(:, 1:r);
V  = Vfull(:, 1:r);

if strcmp(matFormat, 'standard')
    Theta_star = (U * V') / sqrt(r);
elseif strcmp(matFormat, 'factor')
    Theta_star.U = U;
    Theta_star.V = V;
    Theta_star.S = diag(ones(1,r))/sqrt(r);
else
    error('No such data structure defined!');
end

% Theta_star  = (U * V') / sqrt(r);

% Generate a dataset in compressed form
switch sampling
    case 'wr'          % with replacement
        for i = 1 : N
            rowInd = ceil(nr * rand(1));
            colInd = ceil(nc * rand(1));
            % X(row_index, col_index, i) = sqrt(nr*nc);
            % y(i) = Theta_star(row_index, col_index)*sqrt(nr*nc);
            II(i) = rowInd;
            JJ(i) = colInd;
            y(i)  = sum(U(rowInd, :) .* V(colInd, :)) * sqrt(nr*nc/r);
        end
    case 'wor'         % without replacement
        if pfrac >= 1
            error("Too large sampling portion for without-replacement setting!");
        end
        for j = 1:nc
            tmp = rand(nr,1);
            idx = find(tmp < pfrac);
            len = length(idx);
            II(cnt+(1:len)) = idx;
            JJ(cnt+(1:len)) = j*ones(len,1);
            cnt = cnt + len;
        end
        II = II(1:cnt); JJ = JJ(1:cnt); N = cnt;
        for i = 1 : N
            y(i) = sum(U(i, :) .* V(i, :)) * sqrt(nr*nc/r);
        end
    otherwise
        error('No such sampling regime defined!');
end


% add perturbations to the sampled value
switch noise.type
    case 'Gaussian'
        y = y + noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
    case 'Cauchy'
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    case 'Lognormal'
        y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1])) - exp(noise.para/2));
    case 'Chi2'
        y = y + noise.scale .* chi2rnd(noise.para, [N, 1]);
    case 'T'
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    case 'Noiseless'
        y = y;
    otherwise
        error('Undefined noise type!');
end



