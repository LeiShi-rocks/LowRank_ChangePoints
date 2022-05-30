function [U, S, V, svp] = proxsolver_sp(sv, gap, param)

global Thetak Thetakk Grad 

svdtol = param.svdtol;
matFormat = param.matFormat;
beta1  = param.beta1;
beta2  = param.beta2;
nr = param.nr;
nc = param.nc;
lambda_run = param.lambda_run;
tau = param.tau;

options.tol =   svdtol;
threshold_value = lambda_run/tau;
if strcmp(matFormat, 'standard')
    Thetakka = beta1 * Thetakk - beta2 * Thetak;
    G        =   Thetakka - Grad/tau;
    [U,S,V]  =   lansvd(G, sv, 'L', options);
else
    [U,S,V]  =   lansvd('matvecprod','matTvecprod', nr, nc, sv, 'L', options, param);
end

d           =   diag(S);
sd          =   max(d - threshold_value, 0);

% truncation of singular values
d2       =  d - threshold_value; 
ratio    =  d2(1:end-1)./d2(2:end);
idxstart =  5; %% 2009 Oct 30
idxbig   =  find(abs(ratio(idxstart:end)) > gap);

if ~isempty(idxbig)
  idxtruncate = (idxstart-1)+idxbig(1)+1;
else
  idxtruncate = length(find(sd>0)); 
end
normratio = zeros(1,idxtruncate); 
window = 10;
for kk = idxstart:idxtruncate-1
  sd1 = d2(max(1,kk-window):kk); 
  sd2 = d2(kk+1:min(idxtruncate,kk+window+1)); 
  normratio(kk) = mean(sd1)/(mean(sd2)+eps); 
end 
[maxnormratio,maxidx] = max(abs(normratio)); 

if (maxnormratio > gap)
    sd = sd(1:maxidx); 
end

sd   =   sd(1: length(sd)); 
svp  =   length(sd);
U    =   U(:, 1:length(sd));
S    =   diag(sd);
V    =   V(:, 1:length(sd));

       