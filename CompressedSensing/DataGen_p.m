function [X, y, Theta_star] = DataGen_p(nr, nc, N, r, noise, type, signal, p, scale)

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
        y(i) = sum(sum(Theta_star .* X(:,:,i)))+ scale*binornd(1,p,1)*2*(binornd(1,0.5,1)-0.5)*(p>0);
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
 
end

