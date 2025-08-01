function plot_l_curve()
% PLOT_L_CURVE Generates the L-curve and Error-vs-Lambda plots to analyze
% the Tikhonov regularization parameter for hybrid GMRES methods.

%% 1) Set up Test Problem & Parameters
n = 32;
[A, b, x_true] = shaw(n);
B = A';
DeltaM = 1e-5 * randn(size(A'));
maxit = n;
tol = 1e-8;

% Define a range of lambda values to test
lambda_range = logspace(-8, 0, 50);

% Initialize storage for results
res_norms_ab = zeros(size(lambda_range));
sol_norms_ab = zeros(size(lambda_range));
err_norms_ab = zeros(size(lambda_range));

res_norms_ba = zeros(size(lambda_range));
sol_norms_ba = zeros(size(lambda_range));
err_norms_ba = zeros(size(lambda_range));

%% 2) Loop through lambdas and solve
fprintf('Generating L-curve data by solving for a range of lambda values...\n');
for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    
    % --- Hybrid AB-GMRES ---
    [x_hab, err_hab, res_hab, ~] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
    res_norms_ab(i) = res_hab(end); % Final residual norm
    sol_norms_ab(i) = norm(x_hab);
    err_norms_ab(i) = err_hab(end); % Final error norm

    % --- Hybrid BA-GMRES ---
    [x_hba, err_hba, res_hba, ~] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
    res_norms_ba(i) = res_hba(end); % Final residual norm
    sol_norms_ba(i) = norm(x_hba);
    err_norms_ba(i) = err_hba(end); % Final error norm
    
    fprintf('Completed lambda = %.2e\n', lambda);
end

%% 3) Create the plots
fprintf('Generating plots...\n');
figure('Name', 'Regularization Analysis (L-Curve and Error)', 'Position', [100 100 1000 400]);

% --- L-Curve Plot ---
subplot(1, 2, 1);
loglog(res_norms_ab, sol_norms_ab, 'b-o', 'DisplayName', 'hybrid AB');
hold on;
loglog(res_norms_ba, sol_norms_ba, 'r-x', 'DisplayName', 'hybrid BA');
hold off;
grid on;
xlabel('Relative Residual Norm ||b - Ax_{\lambda}|| / ||b||');
ylabel('Solution Norm ||x_{\lambda}||');
title('L-Curve');
legend('Location', 'SouthEast');

% --- Error vs. Lambda Plot ---
subplot(1, 2, 2);
[min_err_ab, idx_ab] = min(err_norms_ab);
[min_err_ba, idx_ba] = min(err_norms_ba);
loglog(lambda_range, err_norms_ab, 'b-o', 'DisplayName', 'hybrid AB');
hold on;
loglog(lambda_range, err_norms_ba, 'r-x', 'DisplayName', 'hybrid BA');
% Mark the minimum error points
loglog(lambda_range(idx_ab), min_err_ab, 'bp', 'MarkerSize', 14, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');
loglog(lambda_range(idx_ba), min_err_ba, 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
hold off;
grid on;
xlabel('Regularization Parameter \lambda');
ylabel('Relative Error Norm ||x_{\lambda} - x_{true}|| / ||x_{true}||');
title('Error vs. Lambda');
legend('Location', 'Best');

end