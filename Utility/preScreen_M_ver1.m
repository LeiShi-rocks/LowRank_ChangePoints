function [p_value, outInfo] = preScreen_M(y, X, startInd, endInd, opts)

%% parameter

N      =  setOpts(opts, 'N',  1000);
nr     =  setOpts(opts, 'nr'  , 50);
nc     =  setOpts(opts, 'nc'  , 50);
Nsum   =  sum(N);
method =  setOpts(opts, 'method', 'S');

p      =  nr*nc;
comb_flag = setOpts(opts, 'comb_flag', 'original');

%% TN

Z = zeros(Nsum, nr*nc);
for i = 1 : Nsum
    Z(i, :) = y(i) * reshape(X(:,:,i), [1, nr*nc]);
end

TN = zeros(endInd - startInd + 1, 1);

switch method
    case 'S'  
        for sepInd = startInd : endInd
            
            NL    =  sepInd;
            NR    =  Nsum - sepInd + 1;
            nL    =  NL - 1;
            nR    =  NR - 1;
            ZL    =  Z(1:sepInd, :);
            ZR    =  Z((sepInd+1):Nsum, :);
            
            ML    =  mean(ZL, 1);
            MR    =  mean(ZR, 1);
            diff_ML_MR = ML - MR;
            
            SL    =  cov(ZL);
            SR    =  cov(ZR);
            S     =  SL/NL + SR/NR;
            diagSL       =  diag(SL);
            diagSR       =  diag(SR);
            diagD        =  diag(S);       % vector
            diagD_inv    =  1./(diagD);
            diagD_sq     =  sqrt(diagD);   % vector
            diagD_sq_inv =  1./(diagD_sq);
            
            % compute qn
            qn           =  ( sum(sum(diff_ML_MR .* diff_ML_MR .* diagD_inv')) - p ) / sqrt(p);
            
            % compute Var(qn)
            RD           =  diagD_sq_inv * diagD_sq_inv';
            R            =  RD .* S;
            tr_R2        =  sum(sum(R.^2));
            tr_Dinv_SL   =  sum(diagSL.*diagD_inv);
            tr_Dinv_SR   =  sum(diagSR.*diagD_inv);
            Var_qn       =  2*tr_R2/p - 2*(tr_Dinv_SL)^2/(p*NL^2*nL) - 2*(tr_Dinv_SR)^2/(p*NR^2*nR);
            
            % compute cpn
            cpn     =  1 + tr_R2/(p^1.5);
            
            % compute TN
            TN(sepInd - startInd + 1) = qn / sqrt(Var_qn * cpn);
        end
        
    case 'B' % for Bai 1996
        for sepInd = startInd : endInd
            
            NL    =  sepInd;
            NR    =  Nsum - sepInd + 1;
            nL    =  NL - 1;
            nR    =  NR - 1;
            ZL    =  Z(1:sepInd, :);
            ZR    =  Z((sepInd+1):Nsum, :);
            
            ML    =  mean(ZL, 1);
            MR    =  mean(ZR, 1);
            diff_ML_MR = ML - MR;
            
            SL    =  cov(ZL);
            SR    =  cov(ZR);
            
            % compute Qn
            Qn       =  ( sum(sum(diff_ML_MR .* diff_ML_MR)) - trace(SL)/NL - trace(SR)/NR ) / sqrt(p);
            
            % compute Var(Qn)
            aL       =  (sum(sum(SL.^2)) - (trace(SL))^2/NL) * nL^2 / (p*(nL-1)*(nL+1));
            aR       =  (sum(sum(SR.^2)) - (trace(SR))^2/NR) * nR^2 / (p*(nR-1)*(nR+1));
            tr_SL_SR =  sum(sum(SL.*SR));
            sigma2Q  =  2*aL/NL^2 + 2*aR/NR^2 + 4*tr_SL_SR/(p*NL*NR);

            % compute TN
            TN(sepInd - startInd + 1) = Qn / sqrt(sigma2Q);
            
        end
        
    otherwise
        error('Not defined!');
end

%% report p-value
p_value_original = 2 * (1 - cdf('Normal', abs(TN), 0, 1));
switch comb_flag
    case 'original'
        p_value = p_value_original;
    case 'cauchy'
        p_value = sum(tan((0.5 - p_value_original)*pi))/(endInd - startInd + 1);
    otherwise
        error('Not defined!');
end

outInfo = struct(...
    'p_value_original', p_value_original,...
    'TN', TN);




