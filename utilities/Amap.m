function y = Amap(Theta, X)
dim = size(X);
if length(dim) == 3
    N = size(X, 3);
    y = zeros(N, 1);
    for i = 1 : N
        y(i) = sum(sum(Theta .* X(:,:,i)));
    end
elseif length(dim) == 2
    Sampling = Theta * X;
    y = reshape(Sampling', [], 1);
else
    error('Not compatible dimension!');
end

end