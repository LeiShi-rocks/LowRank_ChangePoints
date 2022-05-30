function [p_value, y, X, outInfo] = runTrial(startInd, endInd, data_opts, screen_opts)

[y, X, outInfo_data]  =   DataGen_ChangePoints(data_opts);
[p_value, outInfo_result] = preScreen_M(y, X, startInd, endInd, screen_opts);

outInfo.data   = outInfo_data;
outInfo.result = outInfo_result;





