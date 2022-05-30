function [y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts)

% noise   =  noise.type + noise.scale + noise.para
% problem    =  CS or MR
% design  =  design.type + design.para
% cp_opts =  cp_opts.num_cp + cp_opts.pos_cp

% Initialize change points
random_flag = setOpts(cp_opts, 'random_flag', 0);
if random_flag == 1
    threshold_var = rand(1,N); % threshold variables
else
    threshold_var = (1:N)/N; % threshold variables
end

num_seg = setOpts(cp_opts, 'num_seg', 1);  % number of segments; by default 0.
pos_seg = setOpts(cp_opts, 'pos_seg', 0);  % left ends of segments; in the scale of [0,1)

if length(pos_seg)~=num_seg
    error('Incompatible num_seg and pos_seg!');
end

pos_seg = [pos_seg, 1];

signal  = setOpts(cp_opts, 'signal', 'large');


switch problem
    case 'CS'
        % generate X
        X = normrnd(0,1,[nr,nc,N]);
        
        
        % generate Theta_star
        if strcmp(signal, 'large')
            Theta_star = zeros(nr, nc, num_seg);
            for k = 1:num_seg
                [U,~,~]   =  svd(normrnd(0,1,[nr, 100]), 'econ');
                [V,~,~]   =  svd(normrnd(0,1,[nc, 100]), 'econ');
                Theta_star(:,:,k)  =  U(:, 1:r) * V(:, 1:r)' / sqrt(r);
            end
        elseif strcmp(signal, 'small')
            Theta_star = zeros(nr, nc, num_seg);
            [U,~,~]   =  svd(normrnd(0,1,[nr, 100]), 'econ');
            [V,~,~]   =  svd(normrnd(0,1,[nc, 100]), 'econ');
            non_vary  =  1:(r-1);
            for k = 1:num_seg
                indices = [non_vary, r-1+k];
                Theta_star(:,:,k)  =  U(:, indices) * V(:, indices)' / sqrt(r);
            end
        end
        
        
        
        % generate noise
        switch noise.type
            case 'Gaussian'
                epsilon = noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
            case 'Cauchy'
                epsilon = noise.scale .* trnd(noise.para, [N, 1]);
            case 'Lognormal'
                epsilon = noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1])) - exp(noise.para/2));
            case 'T'
                epsilon = noise.scale .* trnd(noise.para, [N, 1]);
            otherwise
                error('Noise type not defined!');
        end
        
        
        % generate y
        y = zeros(N, 1);
        for k = 1:num_seg
            pos_ind = find((pos_seg(k) < threshold_var) & (threshold_var <= pos_seg(k+1)));
            if size(pos_ind, 2) > 0
                for ind = pos_ind
                    y(ind) = sum(sum(X(:,:,ind) .* Theta_star(:,:,k))) + epsilon(ind);
                    % disp([k, Theta_star(1,1,k)]);
                end
            end
        end
        
    case 'MR'
        % Y = Theta * X + epsilon, Theta is nr x nc, X is nc x N
        % generate X
        if strcmp(design.type, 'AR')
            % disp('Hi');
            a = design.para;
            
            rows = repmat((1:nc)', [1, nc]);
            cols = repmat((1:nc) , [nc, 1]);
            if a == 0
                Sigma = eye(nc);
            else
                Amatrix = a*ones(nc);
                Sigma = Amatrix.^(abs(rows-cols));
            end
            X = mvnrnd(zeros(nc, 1), Sigma, N)';
            
        elseif strcmp(design.type, 't')
            df = design.para;
            X  = sqrt((df-2)/df).* trnd(df, [nc, N]);
        else
            X  = normrnd(0, 1, [nc, N]);
        end
        
        
        % generate Theta_star
        if strcmp(signal, 'large')
            Theta_star = zeros(nr, nc, num_seg);
            for k = 1:num_seg
                [U,~,~]   =  svd(normrnd(0,1,[nr, 100]), 'econ');
                [V,~,~]   =  svd(normrnd(0,1,[nc, 100]), 'econ');
                Theta_star(:,:,k)  =  U(:, 1:r) * V(:, 1:r)' / sqrt(r);
            end
        elseif strcmp(signal, 'small')
            Theta_star = zeros(nr, nc, num_seg);
            [U,~,~]   =  svd(normrnd(0,1,[nr, 100]), 'econ');
            [V,~,~]   =  svd(normrnd(0,1,[nc, 100]), 'econ');
            non_vary  =  1:(r-1);
            for k = 1:num_seg
                indices = [non_vary, r-1+k];
                Theta_star(:,:,k)  =  U(:, indices) * V(:, indices)' / sqrt(r);
            end
        end
        
        
        % generate noise
        switch noise.type
            case 'Gaussian'
                epsilon = noise.scale .* normrnd(0, sqrt(noise.para), [nr, N]);
            case 'Cauchy'
                epsilon = noise.scale .* trnd(noise.para, [nr, N]);
            case 'Lognormal'
                epsilon = noise.scale .* (exp(normrnd(0, sqrt(noise.para), [nr, N])) - exp(noise.para/2));
            case 'Chi2'
                epsilon = noise.scale .* chi2rnd(noise.para, [nr, N]);
            case 'T'
                epsilon = noise.scale .* trnd(noise.para, [nr, N])/2;
            otherwise
                error('Noise type not defined!');
        end
        
        % generate y
        y = zeros(nr, N);
        for k = 1:num_seg
            pos_ind = find((pos_seg(k) < threshold_var) & (threshold_var <= pos_seg(k+1)));
            for ind = pos_ind
                y(:, ind) = Theta_star(:,:,k) * X(:, ind) + epsilon(:, ind);
            end
        end
        
    otherwise
        error('No such type designed!');
end


outInfo.Theta_star = Theta_star;
outInfo.threshold_var = threshold_var;
