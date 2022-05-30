function T = ATmap(bb, X) 
dim = size(X);
if length(dim) == 3
    nr = dim(1);
    nc = dim(2);
    N = dim(3);
    T = zeros(nr, nc);
    for i = 1 : N
        T = T + bb(i) * X(:, :, i);
    end
elseif length(dim) == 2
%    nc = dim(1);
    N = dim(2);
    bbM = reshape(bb, N, []); % N x nr
    T = (X * bbM)';
else
    error('Not compatible dimension!');
end


end