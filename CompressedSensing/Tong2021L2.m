function [Theta_hat] = Tong2021L2(X, y, r_est)

dim = size(X);
nr = dim(1);
nc = dim(2);
N = dim(3);
tol = 1e-4;

T = 100;

%% Spectral initialization
Y = zeros(nr, nc);
for k = 1:N
    Y = Y + y(k)*X(:,:,k);
end
[U0, Sigma0, V0] = svds(Y/N, r_est);
%% ScaledSM
L = U0*sqrt(Sigma0);
R = V0*sqrt(Sigma0);
Theta_new= L*R';
%error = zeros(T,1);
for t = 1:T
    Theta_old= Theta_new;
     %error(t) = log(norm(Theta_hat-Theta_star, 'fro'));
     %error(t);
    Z = zeros(nr, nc); 
    for k = 1:N
        z = sum(sum(X(:,:,k).*Theta_old)) - y(k);
        Z = Z + z*X(:,:,k)/N;
    end
   %eta = (lambda*q^t)/sqrt(ZL(:)'*ZLpinv(:) + ZR(:)'*ZRpinv(:));
    %eta = (loss-loss_star)/(ZL(:)'*ZLpinv(:) + ZR(:)'*ZRpinv(:));
    eta = 0.5;
    L = L - eta*Z*R/(R'*R);
    R = R - eta*Z'*L/(L'*L);
    Theta_new = L*R';
    diff = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    if diff < tol
        break;
    end
end
Theta_hat = Theta_new;
end

