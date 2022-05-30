clear; close all; clc;
n = 40;
r = 10;
kappa = 10;
snr_list = 10;
m = 5*n*r;

T = 600;
eta = 0.5;
thresh_up = 1e3; thresh_low = 1e-14;
errors_ScaledGD = zeros(length(snr_list), T);
errors_GD = zeros(length(snr_list), T);

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(U_seed, r);
As = cell(m, 1);
for k = 1:m
	As{k} = randn(n, n)/sqrt(m);
end
noise_seed = randn(m, 1);
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
    y = zeros(m, 1);
    for k = 1:m
        y(k) = As{k}(:)'*X_star(:);
    end
    y = y + norm(X_star, 'fro')/n/snr*noise_seed;
    %% Spectral initialization
    Y = zeros(n, n);
    for k = 1:m
        Y = Y + y(k)*As{k};
    end
    [U0, Sigma0, V0] = svds(Y, r);
    %% Scaled GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledGD(i_snr, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        L_plus = L - eta*Z*R/(R'*R);
        R_plus = R - eta*Z'*L/(L'*L);
        L = L_plus;
        R = R_plus;
    end
end
errors_ScaledGD
z = zeros(N,1);
for i = 1:N
   z(i) = sum(sum(X(:,:,i).*Theta_star))-y(i);
end