clc;
mean_Record = mean(Record, 3);
mean_Record(:,1,:,:,:) = mean_Record(:,1,:,:,:) * 6400;
disp(num2str(mean_Record(:,:,1,1,3)));
disp(num2str(mean_Record(:,:,1,2,3)));
disp(num2str(mean_Record(:,:,1,3,3)));


std_Record = std(Record, 0, 3);
std_Record(:,1,:,:,:) = std_Record(:,1,:,:,:) * 6400;
disp(num2str(std_Record(:,:,1,1,3)));
disp(num2str(std_Record(:,:,1,2,3)));
disp(num2str(std_Record(:,:,1,3,3)));