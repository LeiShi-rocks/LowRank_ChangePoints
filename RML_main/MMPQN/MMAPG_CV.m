function [ClambdaHat, oneSDScore, elapsedTime, CtauHat, oneSDScore_tau] = MMAPG_CV(y, X, type, Clambda_Cand, fold, robust, trim, plot_flag)

% This function is used to perform CV using APGL algorithm
% y, X: response and matrix
% type: struct
% - name: loss to use; see MMAPG
% - para: parameter of the loss, only for Huber
% - eta:  curvature updating constant
% - Lf:   maximal curvature
% Clambda_Cand: Clambda Candidate vector
% fold: number of folds in CV
% robust: struct
% - flag: whether to perform robustifying
% - para: a percentage indicating how much testing data to use when
% calculating score; will be set to 1 if robust.flag == false.
% trim: struct
% - flag: whether to perform trimming; only for robust shrinkage
% --- if this is true then please ctrl-F for vector `quan`(meaning quantile) 
%   and set the quantiles you want to test!
% plot_flag: whether to plot the score.


tol = 1e-4;
maxiter = 100;

l = length(Clambda_Cand);
dim = size(X);
N = dim(3);
nr = dim(1);
nc = dim(2);
K = fold;
CtauHat = Inf;
oneSDScore_tau = Inf;

robustify = robust;
if ~robust.flag
    robustify.para = 1;
end

cvScore = zeros(l,1);

tic;
index = rand(N);
[~, index] = sort(index);

for cvFold = 1 : K % cvFold = 1;
    natVec = rem(1:N, K);
    testInd = sort(index(natVec == (cvFold-1)));
    trainInd = sort(index(natVec ~= (cvFold-1)));
    X_train = X(:,:,trainInd);
    y_train = y(trainInd);
    X_test = X(:,:,testInd);
    y_test = y(testInd);
    if robustify.flag
        disp(['Truncating ', num2str((1-robustify.para)*100), '% of the response...']);
        [~, preTrimmedTestInd] = sort(abs(y_test));
        trimmedTestInd = preTrimmedTestInd(1 : floor(robustify.para*length(y_test)));
        X_test = X_test(:,:,trimmedTestInd);
        y_test = y_test(trimmedTestInd);
    end
    for cv_iter = 1:l
        disp(['cvFold: ', num2str(cvFold), ' cv_iter: ', num2str(cv_iter)]);
        % cv_iter = 1;
        lambda_cv = Clambda_Cand(cv_iter) * sqrt((nr+nc) / N);
        if ~strcmp(type.name, 'APGL')
            [ThetaCV, ~] = MMAPG(y_train, X_train, type, lambda_cv, tol, maxiter, zeros(nr,nc));
        else
            ThetaCV = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X_train),...
                @(bb) ATmap(bb, X_train), y_train, lambda_cv, 0);
        end
        for i = 1:length(y_test)
            cvScore(cv_iter) = cvScore(cv_iter) + (y_test(i) - sum(sum(X_test(:,:,i).*ThetaCV)))^2;
        end
    end
end
cvScore = cvScore/(N*robustify.para);



scoreSD = std(cvScore);
[bestScore, bestInd] = min(cvScore); 
cvScoreSeq = cvScore(bestInd:end);
oneSDScore = bestScore + scoreSD;
bestInd = min(find(cvScoreSeq <= oneSDScore, 1, 'last') + bestInd - 1, l);

ClambdaHat = Clambda_Cand(bestInd);


if trim.flag
    disp('Tuning tau_y...');
    
    lambdaHat = ClambdaHat * sqrt((nr+nc)/N);
    sort_y = sort(abs(y));
    quan = trim.para;
    tau_Cand = sort_y(floor(quan*N));
    
    % Ctau_Cand = tau_Cand/(sqrt(N/(nr+nc)));
    
    cvLength_tau = length(tau_Cand);
    cvScore_tau = zeros(1,cvLength_tau);
    
    index = rand(N);
    [~, index] = sort(index);

    for cvFold = 1 : K
        natVec = rem(1:N, K);
        testInd = index(natVec == cvFold);
        trainInd = index(natVec ~= cvFold);
        X_train = X(:,:,trainInd);
        X_test = X(:,:,testInd);
        y_test = y(testInd);
        if robustify.flag
            disp(['Truncating ', num2str((1-robustify.para)*100), '% of the covariates...']);
            [~, preTrimmedTestInd] = sort(abs(y_test));
            trimmedTestInd = preTrimmedTestInd(1 : floor(robustify.para*length(y_test)));
            X_test = X_test(:,:,trimmedTestInd);
            y_test = y_test(trimmedTestInd);
        end
        for num_tau = 1 : cvLength_tau
            disp(['cvFold: ', num2str(cvFold), ' num_tau: ', num2str(num_tau)]);
            tau_cv = tau_Cand(num_tau);
            y_shrink = sign(y).*min(abs(y), tau_cv);
            y_train = y_shrink(trainInd);
            if ~strcmp(type.name, 'APGL')
                [ThetaCV, ~] = MMAPG(y_train, X_train, type, lambdaHat, tol, maxiter, zeros(nr,nc));
            else
                ThetaCV = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X_train),...
                   @(bb) ATmap(bb, X_train), y_train, lambdaHat, 0);
            end
            for i = 1:length(y_test)
                cvScore_tau(cvFold) = cvScore_tau(cvFold) + (y_test(i) - sum(sum(X_test(:,:,i).*ThetaCV)))^2;
            end
        end
    end
    cvScore_tau = cvScore_tau / (N*robustify.para);

    
    scoreSD_tau = std(cvScore_tau);
    [bestScore_tau, bestInd_tau] = min(cvScore_tau);
    cvScoreSeq_tau = cvScore_tau(1:bestInd_tau);
    oneSDScore_tau = bestScore_tau + scoreSD_tau;
    bestInd_tau = max(find(cvScoreSeq_tau > oneSDScore_tau, 1), 1);
    
    CtauHat = tau_Cand(bestInd_tau) / (sqrt(N/(nr+nc)));
end


if plot_flag 
   if ~trim.flag
       plot(Clambda_Cand * sqrt((nr+nc) / N), cvScore);
   end
   if trim.flag
       subplot(121);
       plot(Clambda_Cand * sqrt((nr+nc) / N), cvScore);
       hold on;
       xline(lambdaHat);
       hold off;
       subplot(122);
       plot(quan, cvScore_tau);
       hold on;
       xline(quan(bestInd_tau));
       hold off;
   end
end

elapsedTime = toc;