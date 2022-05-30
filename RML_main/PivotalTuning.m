function [lambdaHat, elapsedTime] = PivotalTuning(X, alpha, Qtype, noise)
nr = Qtype.nr;
nc = Qtype.nc;
N = Qtype.N;
B =100; % Monte Carlo sample size
if strcmp(Qtype.loss, 'L2')
    
    if strcmp(Qtype.name, 'M')
        
        Grad = zeros(nr, nc);
        ecdf = zeros(1, B);
        
        tic;
        for i = 1 : B
            %disp(['Calculating: ', num2str(i)]);
            if strcmp(noise.type, 'Gaussian')
                weight = noise.scale .* normrnd(0, sqrt(noise.para), [N, 1]);
            elseif strcmp(noise.type, 'Cauchy')
                weight = noise.scale .* trnd(noise.para, [N, 1]);
            elseif strcmp(noise.type, 'Lognormal')
                weight = noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(noise.para/2));
                %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(0));
            elseif strcmp(noise.type, 'T')
                weight = noise.scale .* trnd(noise.para, [N, 1]);
            end
            
            for j = 1 : N
                Grad = Grad + 2*weight(j).* X(:,:,j);
            end
            
            ecdf(i) = norm(Grad);
            Grad = zeros(nr,nc);
        end
        ecdf = sort(ecdf);
        lambdaHat = 1.01 * ecdf(floor(B*(1-alpha)));
        elapsedTime = toc;
    elseif strcmp(Qtype.name, 'V')
        
        Grad = zeros(nr, nc);
        ecdf = zeros(1, B);
        
        tic;
        for i = 1 : B
            %         disp(['Calculating: ', num2str(i)]);
            if strcmp(noise.type, 'Gaussian')
                weight = noise.scale .* normrnd(0, sqrt(noise.para), [nr, N]);
            elseif strcmp(noise.type, 'Cauchy')
                weight = noise.scale .* trnd(noise.para, [nr, N]);
            elseif strcmp(noise.type, 'Lognormal')
                weight = noise.scale .* (exp(normrnd(0, sqrt(noise.para), [nr, N]))-exp(noise.para/2));
                %y = y + noise.scale .* (exp(normrnd(0, sqrt(noise.para), [N, 1]))-exp(0));
            elseif strcmp(noise.type, 'T')
                weight = noise.scale .* trnd(noise.para, [nr, N]);
            end
            
            Grad = 2*weight * X'/nr;
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
elseif strcmp(Qtype.loss, 'L1')
    
    if strcmp(Qtype.name, 'M')
        
        Grad = zeros(nr, nc);
        ecdf = zeros(1, B);
        
        tic;
        for i = 1 : B
            %disp(['Calculating: ', num2str(i)]);
            weight = (2 * (rand(1,N)>0.5) - 1);
            for j = 1 : N
                Grad = Grad + weight(j).* X(:,:,j);
            end
            
            ecdf(i) = norm(Grad);
            Grad = zeros(nr,nc);
        end
        ecdf = sort(ecdf);
        lambdaHat = 1.01 * ecdf(floor(B*(1-alpha)));
        elapsedTime = toc;
    elseif strcmp(Qtype.name, 'V')
        
        Grad = zeros(nr, nc);
        ecdf = zeros(1, B);
        
        tic;
        for i = 1 : B
            %         disp(['Calculating: ', num2str(i)]);
            weight = (2 * (rand(nr,N)>0.5) - 1);

            Grad = weight * X'/nr;
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
elseif strcmp(Qtype.loss, 'wilcoxon')
    if strcmp(Qtype.name, 'M')
        
        Grad = zeros(nr, nc);
        ecdf = zeros(1, B);
        
        tic;
        for i = 1 : B
            %disp(['Calculating: ', num2str(i)]);
            [~, randomRank] = sort(rand(N,1));
            for j = 1 : N
                Grad = Grad + (randomRank(j)/(N-1)-0.5*(N+1)/(N-1)).* X(:,:,j);
            end
            
            ecdf(i) = norm(Grad);
            Grad = zeros(nr,nc);
        end
        ecdf = sort(ecdf);
        lambdaHat = 1.0 * ecdf(floor(B*(1-alpha)));
        elapsedTime = toc;
    elseif strcmp(Qtype.name, 'V')
        
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
    
end