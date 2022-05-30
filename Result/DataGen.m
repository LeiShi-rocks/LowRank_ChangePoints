function [X, y, Theta_star] = DataGen(nr, nc, N, r, noise, type, signal, design)

% signal = 'small';

if strcmp(type, 'CS')
    if strcmp(signal, 'large')
        % Martin
        Theta_star = normrnd(0,1,[nr, r]) * normrnd(0,1,[r, nc]);
    end
    
%     if strcmp(signal, 'small')
%         % Fan
%         [U,~,~] = svd(normrnd(0,1,[max(nr,nc), 100]), 'econ');
%         Theta_star = U(:, 1:r) * U(:, 1:r)';
%     end
    if strcmp(signal, 'small')
        [U,~,~] = svd(normrnd(0,1,[nr, 100]), 'econ');
        [V,~,~] = svd(normrnd(0,1,[nc, 100]), 'econ');
        Theta_star = U(:, 1:r) * V(:, 1:r)';
    end
    X = zeros(nr, nc, N);
    y = zeros(N, 1);
    %Sigma = (ones(nc*nr,nc*nr) + diag(ones(nc*nr, 1)))./2;
    %Sigma = diag(ones(nc*nr, 1));
    for i = 1 : N
        %    disp(['Currently generating sample: ', num2str(i)]);
        X(:,:,i) = normrnd(0, 1, [nr, nc]);
        %X(:,:,i) = reshape(mvnrnd(zeros(nc*nr, 1), Sigma, 1),nr,nc);
        y(i) = sum(sum(Theta_star .* X(:,:,i)));
    end
    if strcmp(noise.type, 'Gaussian')
        y = y + noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
    elseif strcmp(noise.type, 'Cauchy')
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    elseif strcmp(noise.type, 'Lognormal')
        y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(noise.para/2));
        %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(0));
    elseif strcmp(noise.type, 'T')
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    end
elseif strcmp(type, 'MC')
    if strcmp(signal, 'large')
        % Martin
        Theta_star = normrnd(0,1,[nr, r]) * normrnd(0,1,[r, nc]);
    end
    
%     if strcmp(signal, 'small')
%         % Fan
%         [U,~,~] = svd(normrnd(0,1,[max(nr,nc), 100]), 'econ');
%         Theta_star = U(:, 1:r) * U(:, 1:r)'/sqrt(r);
%     end
    if strcmp(signal, 'small')
        [U,~,~] = svd(normrnd(0,1,[nr, 100]), 'econ');
        [V,~,~] = svd(normrnd(0,1,[nc, 100]), 'econ');
        Theta_star = U(:, 1:r) * V(:, 1:r)'/sqrt(r);
    end
    X = zeros(nr, nc, N);
    y = zeros(N, 1);
    
    
    for i = 1 : N
        row_index = ceil(nr * rand(1));
        col_index = ceil(nc * rand(1));
        X(row_index, col_index, i) = sqrt(nr*nc);
        y(i) = Theta_star(row_index, col_index)*sqrt(nr*nc);
    end
    if strcmp(noise.type, 'Gaussian')
        y = y + noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
    elseif strcmp(noise.type, 'Cauchy')
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    elseif strcmp(noise.type, 'Lognormal')
        y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(noise.para/2));
        %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(0));
    elseif strcmp(noise.type, 'Chi2')
        y = y + noise.scale .* chi2rnd(noise.para, [N, 1]);
    elseif strcmp(noise.type, 'T')
        y = y + noise.scale .* trnd(noise.para, [N, 1]);
    end
elseif strcmp(type, 'MR')
    if strcmp(signal, 'large')
        % Martin
        Theta_star = normrnd(0,1,[nr, r]) * normrnd(0,1,[r, nc]);
    end
    
%     if strcmp(signal, 'small')
%         % Fan
%         [U,~,~] = svd(normrnd(0,1,[max(nr,nc), 100]), 'econ');
%         Theta_star = U(:, 1:r) * U(:, 1:r)';
%     end
    
    if strcmp(signal, 'small')
        [U,~,~] = svd(normrnd(0,1,[nr, 100]), 'econ');
        [V,~,~] = svd(normrnd(0,1,[nc, 100]), 'econ');
        Theta_star = U(:, 1:r) * V(:, 1:r)';
    end
    
    if nargin == 7
        
        X = normrnd(0, 1, [nc, N]);
        
    elseif nargin == 8
        % disp('here');
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
            scaling = design.scaling;
            %X = trnd(df, [nr, N]);
            X = sqrt((df-2)/df).* trnd(df, [nc, N]);
            
        else
            error('No such design defined in DataGen.m!');
        end
    else
        error('You have input too many arguments!');
    end
    
    y = Theta_star * X;
    if strcmp(noise.type, 'Gaussian')
        y = y + noise.scale .* normrnd(0, sqrt(noise.para), [nr, N]);
    elseif strcmp(noise.type, 'Cauchy')
        y = y + noise.scale .* trnd(noise.para, [nr, N]);
    elseif strcmp(noise.type, 'Lognormal')
        y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [nr, N]))-exp(noise.para/2));
        %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [nr, N]))-exp(0));
    elseif strcmp(noise.type, 'Chi2')
        y = y + noise.scale .* chi2rnd(noise.para, [nr, N]);
    elseif strcmp(noise.type, 'T')
        y = y + noise.scale .* trnd(noise.para, [nr, N])/2;
    end
    
    
end

