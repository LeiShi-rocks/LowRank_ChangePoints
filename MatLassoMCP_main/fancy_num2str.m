function str = fancy_num2str(avg, sd, avg_digits, sd_digits, threshold)
%FANCY_NUM2STR Summary of this function goes here
%   Detailed explanation goes here

avg_formatSpec = ['$%.', num2str(avg_digits,'%i'), 'f'];
sd_formatSpec  = ['(%.', num2str(sd_digits,'%i'), 'f)$'];
avg_str = num2str(avg, avg_formatSpec);
sd_str  = num2str(sd , sd_formatSpec);

nrow_sd = size(sd_str, 1);
ncol_sd = size(sd_str, 2);

if ~isempty(find(sd<threshold, 1))
    nrow_zero_sd = sum(sd<threshold);
    sd_str(sd<threshold, :) = repmat(['(0)$', blanks(ncol_sd - 4)], [nrow_zero_sd, 1]);
end

str = [avg_str, sd_str];