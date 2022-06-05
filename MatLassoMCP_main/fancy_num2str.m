function str = fancy_num2str(avg, sd, avg_digits, sd_digits, threshold)
%FANCY_NUM2STR Summary of this function goes here
%   Detailed explanation goes here

avg_formatSpec = [' $%.', num2str(avg_digits,'%i'), 'f'];
sd_formatSpec  = ['(%.', num2str(sd_digits,'%i'), 'f)$ '];
avg_str = num2str(round(avg, avg_digits), avg_formatSpec);
sd_str  = num2str(round(sd, sd_digits) , sd_formatSpec);

nrow_sd = size(sd_str, 1);
ncol_sd = size(sd_str, 2);

%fprintf('nrow_sd: %d, ncol_sd: %d \n', nrow_sd, ncol_sd);

if ~isempty(find(sd<threshold, 1))
    nrow_zero_sd = sum(sd<threshold);
    %fprintf('nrow_zero_sd: %d \n', nrow_zero_sd);
    sd_str(sd<threshold, :) = repmat(['(0)$ ', blanks(ncol_sd - 5)], [nrow_zero_sd, 1]);
end

str = [avg_str, sd_str];