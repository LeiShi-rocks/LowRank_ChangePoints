function [Theta_hat] = Tong2021L1(X, y, r_est)

dim = size(X);
nr = dim(1);
nc = dim(2);
N = dim(3);

T = 100;
lambda = 2;
q = 0.95;
tol = 1e-4;
ps = 0.2;
%% Spectral initialization
Y = zeros(nr, nc);
[~, loc_y] = mink(abs(y), ceil(N*(1-ps)));
for k = loc_y'
    Y = Y + y(k)*X(:,:,k);
end
Y = Y/length(loc_y);
[U0, Sigma0, V0] = svds(Y, r_est);
%% ScaledSM
L = U0*sqrt(Sigma0);
R = V0*sqrt(Sigma0);
Theta_new= L*R';
for t = 1:T
    Theta_old= Theta_new;
    loss = 0;
    %loss_star = 0;
    Z = zeros(nr, nc);
    for k = 1:N
        z = sum(sum(X(:,:,k).*Theta_old)) - y(k);
        %z_star = sum(sum(X(:,:,k).*Theta_star)) - y(k);
        loss = loss + abs(z);
        %loss_star = loss_star + abs(z_star);
        Z = Z + sign(z)*X(:,:,k);
    end
    ZL = Z'*L; ZLpinv = ZL/(L'*L);
    ZR = Z*R; ZRpinv = ZR/(R'*R);
   eta = (lambda*q^t)/sqrt(ZL(:)'*ZLpinv(:) + ZR(:)'*ZRpinv(:));
    %eta = (loss-loss_star)/(ZL(:)'*ZLpinv(:) + ZR(:)'*ZRpinv(:));
    L = L - eta*ZRpinv;
    R = R - eta*ZLpinv;
    Theta_new = L*R';
    diff = norm(Theta_new-Theta_old, 'fro')/norm(Theta_old, 'fro');
    if diff < tol
        break;
    end
end
Theta_hat = Theta_new;
end
