Result = zeros(6,6,3); % df, method, indices
p1 = permute(Record, [4 1 2 3]);
for i = 1 : niter
    Result = Result + p1(:,:,:,i)/niter;
end

figure;
p = plot(tn_Cand, log(Result(:,[1 2 3],1)*nr*nc)/2,  'LineWidth', 2.5, 'MarkerSize', 12); 
% ylim([-9 -3.5]); 
legend({'L_2', 'Robust L_2', 'L_1', 'Rank Matrix Lasso'},'Orientation', 'horizontal',...
    'FontSize', 14, 'Location', 'Northwest');
%ylim([-0.3 0.6]);
xlabel('Degree of freedom for t distribution', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Log Frobenius Error', 'FontSize', 20, 'FontWeight', 'bold');
ax = gca;
set(ax, 'FontSize', 13, 'FontWeight', 'bold');
set(p(1), 'Marker', 'o', 'LineStyle', '-');
set(p(2), 'Marker', '*', 'LineStyle', '--');
set(p(3), 'Marker', 'x', 'LineStyle', '-.');
set(p(4), 'Marker', 'square', 'LineStyle', ':');
axis square;