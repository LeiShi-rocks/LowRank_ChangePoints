function [lambdaHat, oneSDScore, tauHat] = RCVShrinkage(y, X, problemType, lambdaCand, tauCand, yShrinkage, alg, cvPlot)

% This is an implementation of Fan2016Shrinkage.
% ThetaHat: Estimated Theta
% lambdaHat, tauHat: estimated lambda and tau by RCV(robust cv)
% 
% y , X:       response and covariate matrix.
% problemType: four choices, 
%                LM: linear regression;
%                CS: compressed sensing;
%                ML: multivariate regression
%                MC: matrix completion
% lambdaCand:  candidates for lambda in cross validation
% tauCand:     candidates for tauy in cross validation
% yShrinkage:  if heavy tail noise then true; if light tail then false.
% xShrinkage:  if heavy tail design then true; otherwise false.
% alg: algorithm used: 'PRSM' or 'APGL'; only 'APGL' implemented now.
% cvPlot: logical, whether to generate a cv plot.

if ~strcmp(alg, 'APGL')
    error('Not implemented yet!');
end

if strcmp(problemType, 'LM')
    yShrink = @(y, tau) yShrinkage .* sign(y) .* (min(abs(y), abs(tau))) + (1-yShrinkage) .* y; 
    %xShrink = @(x, tau) xShrinkage.*sign(x).*(min(abs(x), abs(tau))) + (1-xShrinkage).*x;
elseif strcmp(problemType, 'ML')
%     if xShrinkage && yShrinkage
%         yShrink = @(y, tau) min(norm(y, 4), abs(tau))/norm(y, 4) * y;
%         %xShrink = @(x, tau) min(norm(x, 4), abs(tau))/norm(x, 4) * x;  
%     else
        yShrink = @(y, tau) yShrinkage .* min(norm(y, 2), abs(tau))/norm(y, 2) .* y + (1-yShrinkage) .* y;
%        xShrink = @(x, tau) x;
%    end
elseif strcmp(problemType, 'CS')
    yShrink = @(y, tau) yShrinkage .* sign(y) .* (min(abs(y), abs(tau))) + (1-yShrinkage) .* y;
%    xShrink = @(x, tau) xShrinkage.*sign(x).*(min(abs(x), abs(tau))) + (1-xShrinkage).*x;
elseif strcmp(problemType, 'MC')
    yShrink = @(y, tau) yShrinkage .* sign(y) .* (min(abs(y), abs(tau))) + (1-yShrinkage) .* y;
%    xShrink = @(x, tau) xShrinkage.*sign(x).*(min(abs(x), abs(tau))) + (1-xShrinkage).*x;
else
    error('This model has not been implemented so far!');
end


dim = size(X);
% Initialization:
nr = dim(1);
nc = dim(2);
N = dim(3);
lambdaHat = lambdaCand(1);
tauHatx = Inf;
tauHaty = Inf;
K = 5; % K folds CV;
eta = 0.95; % trimming percentage


%% Robust cross validation for choosing lambdaHat, tauHatx, tauHaty
% Tune lambda
disp('Tuning lambda...');
cvLength_lambda = length(lambdaCand);
cvScore_lambda = zeros(1,cvLength_lambda);
cvRec_lambda = zeros(1,K);

index = rand(N);
[~, index] = sort(index);
% X_cv = X(:,:,index);
% y_cv = y(index);

for num_lambda = 1 : cvLength_lambda
%    disp(['num_lambda: ', num2str(num_lambda)]);
    lambda_cv = lambdaCand(num_lambda);
    for cvFold = 1 : K
        disp(['num_lambda: ', num2str(num_lambda), ' cvFold: ', num2str(cvFold)]);
        natVec = rem(1:N, K);
        testInd = sort(index(natVec == cvFold));       
        trainInd = sort(index(natVec ~= cvFold));
        X_train = X(:,:,trainInd);
        y_train = y(trainInd);
        X_test = X(:,:,testInd);
        y_test = y(testInd);
        ThetaCV = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X_train), @(bb) ATmap(bb, X_train), y_train, lambda_cv, 0);
        [~, preTrimmedTestInd] = sort(abs(y_test));
        trimmedTestInd = preTrimmedTestInd(1 : floor(eta*length(y_test)));
        X_testTrimmed = X_test(:,:,trimmedTestInd);
        y_testTrimmed = y_test(trimmedTestInd);
        for i = 1:length(trimmedTestInd)
            cvRec_lambda(cvFold) = cvRec_lambda(cvFold) + (y_testTrimmed(i) - sum(sum(X_testTrimmed(:,:,i).*ThetaCV)))^2;
        end
        cvRec_lambda(cvFold) = cvRec_lambda(cvFold) / length(trimmedTestInd);
    end
    cvScore_lambda(num_lambda) = mean(cvRec_lambda);   
end

scoreSD_lambda = std(cvScore_lambda);
[bestScore, bestInd] = min(cvScore_lambda);
cvScoreSeq_lambda = cvScore_lambda(bestInd:end);
oneSDScore = bestScore + scoreSD_lambda;
bestInd = find(cvScoreSeq_lambda > oneSDScore, 1) + bestInd - 1;

lambdaHat = lambdaCand(bestInd);


% Tune tau_x

% Tune tau_y
disp('Tuning tau_y...');
if ~yShrinkage
    tauHat = Inf;
else
    cvLength_tau = length(tauCand);
    cvScore_tau = zeros(1,cvLength_tau);
    cvRec_tau = zeros(1,K);

    index = rand(N);
    [~, index] = sort(index);
%     X_cv = X(:,:,index);
%     y_cv = y(index);

    for num_tau = 1 : cvLength_tau
        tau_cv = tauCand(num_tau);
        y_shrink = yShrink(y, tau_cv);
        for cvFold = 1 : K
            disp(['num_tau: ', num2str(num_tau), ' cvFold: ', num2str(cvFold)]);
            natVec = rem(1:N, K);
            testInd = index(natVec == cvFold);       
            trainInd = index(natVec ~= cvFold);
            X_train = X(:,:,trainInd);
            y_train = y_shrink(trainInd);
            X_test = X(:,:,testInd);
            y_test = y_shrink(testInd);
            ThetaCV = APGL(nr, nc, 'NNLS', @(Theta) Amap(Theta, X_train), @(bb) ATmap(bb, X_train), y_train, lambdaHat, 0);
            [~, preTrimmedTestInd] = sort(abs(y_test));
            trimmedTestInd = preTrimmedTestInd(1 : floor(eta*length(y_test)));
            X_testTrimmed = X_test(:,:,trimmedTestInd);
            y_testTrimmed = y_test(trimmedTestInd);
            for i = 1:length(trimmedTestInd)
                cvRec_tau(cvFold) = cvRec_tau(cvFold) + (y_testTrimmed(i) - sum(sum(X_testTrimmed(:,:,i).*ThetaCV)))^2;
            end
            cvRec_tau(cvFold) = cvRec_tau / length(trimmedTestInd);
        end
        cvScore_tau(num_tau) = mean(cvRec_tau);   
    end

    scoreSD_tau = std(cvScore_tau);
    [bestScore, bestInd] = min(cvScore_tau);
    cvScoreSeq_tau = cvScore_tau(bestInd:end);
    oneSDScore = bestScore + scoreSD_tau;
    bestInd = find(cvScoreSeq_tau > oneSDScore, 1) + bestInd - 1;

    tauHat = tauCand(bestInd);
end

if cvPlot
    subplot(121);
    plot(cvScore_lambda);
    subplot(122);
    plot(cvScore_tau);
end



















