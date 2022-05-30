% Linear Regression Tuning

save_flag = true;

niter = 100;
nr = 1;
nc = 400;
N = 100;

Record = zeros(3, 5, niter, 3); % 3 methods, 3 indices(loss, sparsity, tuning+computation+total time), niter, 3 noise

Theta_star = zeros(nr, nc);
Theta_star(1:3) = ones(nr, 3) * sqrt(3);

noise(1).type = 'Gaussian';
noise(1).para = 1;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1;

% noise(3).type = 'Lognormal';
% noise(3).para = 9;
% noise(3).scale = 1/100;

noise(3).type = 't';
noise(3).para = 4;
noise(3).scale = sqrt(2);

running = [1 1 1];

rng(10, 'twister');

for noiseInd = 1 : 3
    for iter = 1 : niter
        X = normrnd(0,1, [nc, N]);
        y = Theta_star * X;
        if strcmp(noise(noiseInd).type, 'Gaussian')
            y = y + noise(noiseInd).scale .* normrnd(0, sqrt(noise(noiseInd).para), [1, N]);
        elseif strcmp(noise(noiseInd).type, 'Cauchy')
            y = y + noise(noiseInd).scale .* trnd(noise(noiseInd).para, [1, N]);
        elseif strcmp(noise(noiseInd).type, 'Lognormal')
            y = y + noise(noiseInd).scale .* (exp(normrnd(0, sqrt(noise(noiseInd).para), [1, N]))-exp(noise(noiseInd).para/2));
        elseif strcmp(noise(noiseInd).type, 't')
            y = y + noise(noiseInd).scale .* trnd(noise(noiseInd).para, [1, N]);
        end
        
        if running(1)
            tic;           
            [~, fitInfo] = lasso(X', y', 'CV', 5);
            lambdaHat1 = fitInfo.Lambda1SE;
            elapsedTime1 = toc;
            tic;
            Thetahat_1 = lasso(X', y', 'Lambda', lambdaHat1);
            elapsedTime2 = toc;
            Record(1, 1, iter, noiseInd) = norm(Thetahat_1-Theta_star');
            Record(1, 2, iter, noiseInd) = length(find(abs(Thetahat_1)>1e-3));
            Record(1, 3, iter, noiseInd) = elapsedTime1;
            Record(1, 4, iter, noiseInd) = elapsedTime2;
            Record(1, 5, iter, noiseInd) = elapsedTime1 + elapsedTime2;
            disp(' ');
            disp(['Lasso, ', ' noiseInd: ', num2str(noiseInd), ' iter: ', num2str(iter)]);
            disp(' ');
        end
        
        if running(2)
            tic;
            Qtype.name = 'V';
            Qtype.nr = nr;
            Qtype.nc = nc;
            Qtype.N = N;
            lambdaHat2 = PivotalTuning(X, 0.95, Qtype);
            elapsedTime1 = toc;
            tic;
            [Thetahat_2, sparsity] = LP_WilReg(y, X, 2*lambdaHat2/N);
            elapsedTime2 = toc;
            Record(2, 1, iter, noiseInd) = norm(Thetahat_2-Theta_star');
            Record(2, 2, iter, noiseInd) = length(find(abs(Thetahat_2)>1e-3));
            Record(2, 3, iter, noiseInd) = elapsedTime1;
            Record(2, 4, iter, noiseInd) = elapsedTime2;
            Record(2, 5, iter, noiseInd) = elapsedTime1 + elapsedTime2;
            disp(' ');
            disp(['LP_Wilcoxon, ', ' noiseInd: ', num2str(noiseInd), ' iter: ', num2str(iter)]);
            disp(' ');
        end
        
        if running(3)
            tic;
            Qtype.name = 'V';
            Qtype.nr = nr;
            Qtype.nc = nc;
            Qtype.N = N;
            lambdaHat3 = PivotalTuning(X, 0.95, Qtype);
            type.name = 'Wilcoxon';
            type.eta = 0.9;
            type.Lf = 7e2;
            elapsedTime1 = toc;
            tic;
            Thetahat_3 = LRAPG(y, X, type, lambdaHat3*0.5, 1e-4, 100, zeros(nr,nc));
            elapsedTime2 = toc;
            Record(3, 1, iter, noiseInd) = norm(Thetahat_3-Theta_star);
            Record(3, 2, iter, noiseInd) = length(find(abs(Thetahat_3)>1e-34));
            Record(3, 3, iter, noiseInd) = elapsedTime1;
            Record(3, 4, iter, noiseInd) = elapsedTime2;
            Record(3, 5, iter, noiseInd) = elapsedTime1 + elapsedTime2;
            disp(' ');
            disp(['APG_Wilcoxon, ', 'noiseInd: ', num2str(noiseInd), ' iter: ', num2str(iter)]);
            disp(' ');
        end
    end
end


if save_flag
    Record_LR_Tuning = Record;
    save('Record_LR_Tuning.mat', 'Record_LR_Tuning');
end




