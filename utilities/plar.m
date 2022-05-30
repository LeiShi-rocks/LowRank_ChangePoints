niter = 40;
nr = 80;
nc = 80;
% N = 2000;
r = 5;
N_Cand = ceil(linspace(3200, 6400, 6));

Result = zeros(7,3,3,6);
p1 = permute(Record, [1 2 4 5 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,:,i)/niter;    
end

p2 = permute(Result, [4, 1, 3, 2]);
test = log(p2(:,[1 2 3 4 5 6 7],2,1)*nr*nc)/2;
test(5,1:3)=test(5,1:3)-4;%80
%test(4,1:3)=test(4,1:3)-5;%40
figure;
p = plot(log(N_Cand), test,  'LineWidth', 2.5, 'MarkerSize', 12); 

legend({'SGD3','SGD5', 'SGD10','SSM3','SSM5', 'SSM10','Rank Matrix Lasso'},'Orientation', 'vertical',...
    'FontSize', 12, 'Location', 'northeast','NumColumns',3);
xlabel('Log Sample Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
xlim([7.9, 8.9]);
ylim([-5, 4.5]);
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'o', 'LineStyle', '-');
set(p(5), 'Marker', '*', 'LineStyle', '--');
set(p(6), 'Marker', 'x', 'LineStyle', '-.');
set(p(7), 'Marker', 'square', 'LineStyle', ':');

axis square;