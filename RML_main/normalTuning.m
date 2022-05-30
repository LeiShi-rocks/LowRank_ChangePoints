function [lambdaHat, elapsedTime] = normalTuning(X, alpha, Qtype)

if strcmp(Qtype.name, 'M')
    nr = Qtype.nr;
    nc = Qtype.nc;
    N = Qtype.N;
    B = 200; % Monte Carlo sample size
    
    Grad = zeros(nr, nc);
    ecdf = zeros(1, B);
    
    tic;
    for i = 1 : B
        %disp(['Calculating: ', num2str(i)]);
        %[~, randomRank] = sort(rand(N,1));
        z = normrnd(0, 0.5, [1, N]);
        for j = 1 : N
            Grad = Grad + z(j).* X(:,:,j);
        end
        
        ecdf(i) = norm(Grad);
        Grad = zeros(nr,nc);
    end
    ecdf = sort(ecdf);
    lambdaHat = 1.01 * ecdf(floor(B*(1-alpha)));
    elapsedTime = toc;
elseif strcmp(Qtype.name, 'V')
    nr = Qtype.nr;
    nc = Qtype.nc;
    N = Qtype.N;
    B = 100; % Monte Carlo sample size
    
    Grad = zeros(nr, nc);
    ecdf = zeros(1, B);
    
    tic;
    for i = 1 : B
%         disp(['Calculating: ', num2str(i)]);
        [~, randomRank] = sort(rand(nr, N), 2); 
        weight = (randomRank/(N-1)-0.5*(N+1)/(N-1))/nr;
        Grad = weight * X';
        if nr == 1
            ecdf(i) = max(abs(Grad));
        else
            ecdf(i) = norm(Grad);
        end
    end
    ecdf = sort(ecdf);
    lambdaHat = 1.01 * ecdf(floor(B*(1-alpha)));
    elapsedTime = toc;
else
    error('Such type of question has not beed defined yet! Please do that by yourself if you are interested:)');
end