% sparse tutorial

%% construct sparse matrices

A = sparse(10, 10); 
A(5,6) = 3;

II = [1,3,7];
JJ = [2,4,6];
value = [1.4, 2,5, 6.7];
B  = sparse(II, JJ, value);

II = [1,3,7];
JJ = [2,4,6];
value = [1.4, 2,5, 6.7];
C  = sparse(II, JJ, value, 10, 10);

% when there are repeated positions, the values will be added
CC = sparse([1,2], [1,2], [1,2]);

G  = sparse(eye(10));



%% check if a matrix is sparse; desparsify a matrix
issparse(A);
GG = full(G);

%% nearly all built-in functions work well for sparse matrices

% arithmetic calculation
D  =  A + C;
issparse(D);

% certain norm
norm(C, 'fro');
% but operator norm does not work!! 'norm(C)' reports an error.
% svd does not work; instead use svds.

%% for certain functions, sparse matrices rely on special implementation

% operator norm using power iteration
normest(C);

%% calculation involving both sparse and dense matrices

E = ones(10);

F = E + D;

issparse(F);