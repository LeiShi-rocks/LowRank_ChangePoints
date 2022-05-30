function [ThetaHat, sparsity, elapsedTime] = LP_WilReg(y, X, lambda)

B = 500; % rounds of Monte Carlo simulation
dimX = size(X);
N = dimX(2);
nc = dimX(1);

tic;
M = ones(N);
Mtri = triu(M, 1);
[row, col] = find(Mtri);

[~, Ind] = sort(rand([1, N*(N-1)/2]));
Ind = Ind(1:B);
rowMC = row(Ind);
colMC = col(Ind);
Xij_MC = X(:, rowMC)' - X(:, colMC)';
yij_MC = y(rowMC)' - y(colMC)';

f = [ones(B,1)/B; ones(B,1)/B; ones(nc,1)*lambda; zeros(nc,1)];

Aeq = [eye(B), -eye(B), zeros(B, nc), Xij_MC];

beq = yij_MC;

A = [-eye(B), zeros(B), zeros(B,nc), zeros(B,nc);
    zeros(B), -eye(B), zeros(B,nc), zeros(B,nc);
    zeros(nc,B), zeros(nc,B), -eye(nc), eye(nc);
    zeros(nc,B), zeros(nc,B), -eye(nc), -eye(nc)];
    
b = zeros(2*B+2*nc, 1);

x = linprog(f,A,b,Aeq,beq);
         
ThetaHat = x((2*B+nc+1):end);
sparsity = length(find(abs(ThetaHat)>1e-3));
elapsedTime = toc;





