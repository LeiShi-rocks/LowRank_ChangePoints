function [U, S, V, r] = proxsolver(G, gap, lambda_run)

[U,S,V] = svd(G, 'econ');
d = diag(S);
sd = max(d-lambda_run, 0);

% truncation of singular values
d2 = d-lambda_run; 
ratio = d2(1:end-1)./d2(2:end);
idxstart = 5; %% 2009 Oct 30
idxbig = find(abs(ratio(idxstart:end)) > gap);   
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

sd = sd(1: length(sd)); 
r = length(sd);
U = U(:, 1:length(sd));
S = diag(sd);
V = V(:, 1:length(sd));

       