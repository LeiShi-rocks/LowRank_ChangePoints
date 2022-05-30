function [lambdaHat, elapsedTime] = PivotalTuning_spMC(II, JJ, opts)

nr     =  setOpts(opts, 'nr', 100);
nc     =  setOpts(opts, 'nc', 100);
N      =  setOpts(opts, 'N' , 1e3);
B      =  setOpts(opts, 'B' , 100);
alpha  =  setOpts(opts, 'alpha', 0.20);
mtd    =  setOpts(opts, 'mtd', 'PowerIteration');
svdtol =  setOpts(opts, 'svdtol', 1e-6);
verbose = setOpts(opts, 'verbose', 0);
loss   =  setOpts(opts, 'loss', 'Wilcoxon');

noise_dft =  struct(...
    'type' , 'Gaussian',...
    'scale', 1.0,...
    'para' , 0.25);
noise     =  setOpts(opts, 'noise', noise_dft);

sqrt_nr_nc = sqrt(nr*nc);

% sparse gradient initialization
spGrad = zeros(N,1);
Grad   = sparse(nr, nc);
ecdf   = zeros(1, B);

if verbose
    fprintf('\n\n Pivotal tuning(sparse): \n');
end

tic;
switch loss
    case 'Wilcoxon'
        for i = 1 : B
            if verbose
                perc = ceil(0.02 * B);
                % disp(['------- Calculating: ', num2str(i), ' ------- ']);
                if mod(i, perc) == 0
                   fprintf('*');
                end
                if mod(i, 5*perc) == 0 || i==B
                    fprintf('%1.0d/%1.0d|', i, B);
                end
            end
            [~, randomRank] = sort(rand(N,1));
            wt  =  (randomRank./(N-1)) - (0.5*(N+1)/(N-1));
            
            spGrad = wt * sqrt_nr_nc;
            
            
            Grad  =  sparse(II, JJ, spGrad, nr, nc);
            
            switch mtd
                case 'PowerIteration'  % good for small size matrix
                    [res, cnt] = normestplus(Grad, svdtol); % power iteration
                    ecdf(i) = res;
                    if verbose
                        disp(['Iterations: ', num2str(cnt)]);
                    end
                case 'PROPACK'   % very good for large matrix; also works for sparse matrix
                    options.tol = svdtol;
                    ecdf(i) = lansvd(Grad, 1, 'L', options); % SVD then find the largest eigenvalue; seems slow in experiments
                case 'svds'  % really slow :(
                    ecdf(i) = svds(Grad, 1, 'largest', 'Tolerance', svdtol);
                otherwise
                    error('Not implemented!');
            end

        end
    case 'L2'
        for i = 1 : B
            if verbose
                perc = ceil(0.02 * B);
                % disp(['------- Calculating: ', num2str(i), ' ------- ']);
                if mod(i, perc) == 0
                   fprintf('*');
                end
                if mod(i, 5*perc) == 0 || i==B
                    fprintf('%1.0d/%1.0d|', i, B);
                end
            end
            if strcmp(noise.type, 'Gaussian')
                wt = noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
            elseif strcmp(noise.type, 'Cauchy')
                wt = noise.scale .* trnd(noise.para, [N, 1]);
            elseif strcmp(noise.type, 'Lognormal')
                wt = noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(noise.para/2));
                %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(0));
            elseif strcmp(noise.type, 'T')
                wt = noise.scale .* trnd(noise.para, [N, 1]);
            end
            
            
            spGrad = 2 * wt * sqrt_nr_nc;
            
            
            Grad  =  sparse(II, JJ, spGrad, nr, nc);
            
            switch mtd
                case 'PowerIteration'  % good for small size matrix
                    [res, cnt] = normestplus(Grad, svdtol); % power iteration
                    ecdf(i) = res;
                    if verbose
                        disp(['Iterations: ', num2str(cnt)]);
                    end
                case 'PROPACK'   % very good for large matrix; also works for sparse matrix
                    options.tol = svdtol;
                    ecdf(i) = lansvd(Grad, 1, 'L', options); % SVD then find the largest eigenvalue; seems slow in experiments
                case 'svds'  % really slow :(
                    ecdf(i) = svds(Grad, 1, 'largest', 'Tolerance', svdtol);
                otherwise
                    error('Not implemented!');
            end
        end
end
elapsedTime = toc;

ecdf = sort(ecdf);
lambdaHat = 1.0 * ecdf(floor(B*(1-alpha)));

if verbose
    fprintf('\n Done! lambdaHat = %5.4e; Time = %3.2d. \n', lambdaHat, elapsedTime);
end
