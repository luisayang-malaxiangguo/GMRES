function plot_filter_factors()
% Generates two figures:
% 1. A 2x2 subplot comparing the FINAL theoretical vs. empirical filter factors.
% 2. A combined plot showing the EVOLUTION of theoretical filter factors at k=2, 10, and 20.

%% 1) Set up Test Problem
n = 32;
% Select the problem to run. 'deriv2' matches the paper's figures.
 [A, b, x_true] = deriv2(n);
%[A, b, x_true] = shaw(n);
%[A, b, x_true] = heat(n);

%% 2) Algorithm Parameters
tol     = 1e-6;
maxit   = n;      % Run to full dimension
lambda  = 1e-3;
B       = A';     % Matched back-projector
% Use a fixed seed for the random perturbation for reproducibility
rng(0); 
DeltaM  = 1e-5 * randn(size(A'));

%% 3) Run Each Method & Collect Full Iterative History
fprintf('Running GMRES variants...\n');

[x_ab, err_ab, res_ab, it_ab, phi_ab_final, dphi_ab_final, phi_ab_iter,  dphi_ab_iter] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
[x_ba, err_ba, res_ba, it_ba, phi_ba_final, dphi_ba_final, phi_ba_iter,  dphi_ba_iter] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
[x_hab, err_hab, res_hab, it_hab, phi_hab_final, dphi_hab_final, phi_hab_iter, dphi_hab_iter] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
[x_hba, err_hba, res_hba, it_hba, phi_hba_final, dphi_hba_final, phi_hba_iter, dphi_hba_iter] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);

fprintf('All methods complete.\n');

%% 4) Plot 1: Final Theoretical vs. Empirical Factors (2x2 Subplot)
fprintf('Generating final filter factor comparison plot...\n');

% Compute empirical filters  
[U,S,V] = svd(A,'econ');
sigma   = diag(S);
d       = U' * b;

% Ensure division by d is safe for near-zero values
d(abs(d) < 1e-12) = 1; 

Phi_emp_ab  = sigma .* (V' * x_ab) ./ d;
Phi_emp_ba  = sigma .* (V' * x_ba) ./ d;
Phi_emp_hab = sigma .* (V' * x_hab) ./ d;
Phi_emp_hba = sigma .* (V' * x_hba) ./ d;

figure('Name', 'Final Filter Factor Comparison', 'Position', [100 100 800 600]);

% Subplot 1: non-hybrid AB
subplot(2,2,1);
% CORRECTED: Use the correct variable 'phi_ab_final'
kmin = min(numel(phi_ab_final), numel(Phi_emp_ab));
plot(1:kmin, real(phi_ab_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_ab(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('AB-GMRES (non-hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 2: non-hybrid BA
subplot(2,2,2);
kmin = min(numel(phi_ba_final), numel(Phi_emp_ba));
plot(1:kmin, real(phi_ba_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_ba(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('BA-GMRES (non-hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 3: hybrid AB
subplot(2,2,3);
kmin = min(numel(phi_hab_final), numel(Phi_emp_hab));
plot(1:kmin, real(phi_hab_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_hab(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('AB-GMRES (hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 4: hybrid BA
subplot(2,2,4);
kmin = min(numel(phi_hba_final), numel(Phi_emp_hba));
plot(1:kmin, real(phi_hba_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_hba(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('BA-GMRES (hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

%% Plot 2: Combined Evolution of Theoretical Factors
% Theoretical Filter Factors at k
k_values = [2, 16, 32];
figure('Name', 'Evolution of Theoretical Filter Factors', 'Position', [100 100 1200 450]);

t = tiledlayout(1, length(k_values), 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Evolution of Theoretical Filter Factors for Different Iterations (k)', 'FontSize', 14);

% Loop over the k-values to create each subplot
for i = 1:length(k_values)
    k = k_values(i);
    
    nexttile; 
    hold on;
    lw = 1.6;

    % Plot each method for the current k
    if k <= it_ab
        plot(1:k, real(phi_ab_iter{k}), '--', 'LineWidth', lw, 'DisplayName', 'non-hybrid AB');
    end
    if k <= it_ba
        plot(1:k, real(phi_ba_iter{k}), ':', 'LineWidth', lw, 'DisplayName', 'non-hybrid BA');
    end
    if k <= it_hab
        plot(1:k, real(phi_hab_iter{k}), '-', 'LineWidth', lw, 'DisplayName', 'hybrid AB');
    end
    if k <= it_hba
        plot(1:k, real(phi_hba_iter{k}), '-.', 'LineWidth', lw, 'DisplayName', 'hybrid BA');
    end
    
    hold off;
    grid on;
    xlabel('Mode index i');
    ylabel('Filter factor \phi_{i,k}');
    title(sprintf('k = %d', k)); 
    legend('Location', 'Best');
    ylim([-0.2, 1.2]); 
end

%% Plot 3: Evolution for Each Method
k_values = [2, 16, 32];
figure('Name', 'Filter Factor Evolution per Method', 'Position', [100 100 900 700]);

t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Evolution of Theoretical Filter Factors for Each GMRES Variant', 'FontSize', 14);

% --- Plot 1: non-hybrid AB ---
nexttile;
hold on;
for k = k_values
    if k <= it_ab
        plot(1:k, real(phi_ab_iter{k}), '.-', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', sprintf('k = %d', k));
    end
end
hold off; grid on;
title('non-hybrid AB-GMRES');
xlabel('Mode index i'); ylabel('Filter factor \phi_{i,k}');
legend('Location', 'Best');
ylim([-0.5, 1.5]);

% --- Plot 2: non-hybrid BA ---
nexttile;
hold on;
for k = k_values
    if k <= it_ba
        plot(1:k, real(phi_ba_iter{k}), '.-', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', sprintf('k = %d', k));
    end
end
hold off; grid on;
title('non-hybrid BA-GMRES');
xlabel('Mode index i');
legend('Location', 'Best');
ylim([-0.5, 1.5]);

% --- Plot 3: hybrid AB ---
nexttile;
hold on;
for k = k_values
    if k <= it_hab
        plot(1:k, real(phi_hab_iter{k}), '.-', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', sprintf('k = %d', k));
    end
end
hold off; grid on;
title('hybrid AB-GMRES');
xlabel('Mode index i'); ylabel('Filter factor \phi_{i,k}');
legend('Location', 'Best');
ylim([-0.5, 1.5]);

% --- Plot 4: hybrid BA ---
nexttile;
hold on;
for k = k_values
    if k <= it_hba
        plot(1:k, real(phi_hba_iter{k}), '.-', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', sprintf('k = %d', k));
    end
end
hold off; grid on;
title('hybrid BA-GMRES');
xlabel('Mode index i');
legend('Location', 'Best');
ylim([-0.5, 1.5]);

%% Magnitude of the Perturbation Bound
fprintf('Generating perturbation bound magnitude plot...\n');

figure('Name', 'Magnitude of Perturbation Bounds', 'Position', [100 100 900 700]);
t_bounds = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t_bounds, 'Magnitude of Perturbation Bounds $|\delta\phi_{i,k}|$', ...
      'FontSize', 14, 'Interpreter', 'latex');

colors = lines(length(k_values));

%  non-hybrid AB
ax1 = nexttile;
plot_bound_magnitudes_tile(ax1, dphi_ab_iter, it_ab, k_values, colors, 'non-hybrid AB-GMRES');
ylabel(ax1, 'Bound Magnitude $|\delta\phi_{i,k}|$', 'Interpreter', 'latex');

%  non-hybrid BA
ax2 = nexttile;
plot_bound_magnitudes_tile(ax2, dphi_ba_iter, it_ba, k_values, colors, 'non-hybrid BA-GMRES');

% hybrid AB
ax3 = nexttile;
plot_bound_magnitudes_tile(ax3, dphi_hab_iter, it_hab, k_values, colors, 'hybrid AB-GMRES');
ylabel(ax3, 'Bound Magnitude $|\delta\phi_{i,k}|$', 'Interpreter', 'latex');

% hybrid BA
ax4 = nexttile;
plot_bound_magnitudes_tile(ax4, dphi_hba_iter, it_hba, k_values, colors, 'hybrid BA-GMRES');

end

% --- Helper function for plotting bound MAGNITUDES ---
function plot_bound_magnitudes_tile(ax, dphi_iter, iters, k_vals, colors, plot_title)
    % This local function plots the MAGNITUDE of the perturbation bounds on a log scale.
    hold(ax, 'on');
    for i = 1:length(k_vals)
        k = k_vals(i);
        if k <= iters
            % Get the magnitude of the bound. Add eps to prevent log(0) warnings.
            dphi_k_mag = abs(real(dphi_iter{k})) + eps; 
            indices = (1:k)';
            
            % Use SEMILOGY to plot the magnitude of the bound
            semilogy(ax, indices, dphi_k_mag, '.-', 'Color', colors(i,:), 'LineWidth', 1.5, ...
                     'MarkerSize', 12, 'DisplayName', sprintf('k = %d', k));
        end
    end
    hold(ax, 'off');
    grid on;
    title(ax, plot_title);
    xlabel(ax, 'Mode index i');
    ylabel(ax, 'Bound Magnitude $|\delta\phi_{i,k}|$', 'Interpreter', 'latex');
    legend(ax, 'Location', 'Best');
end

%% Plot 5: Unified Convergence History

figure('Name', 'Unified Convergence History', 'Position', [100 100 1000 400]);
lw = 1.8;

% Subplot for Relative Error
subplot(1, 2, 1);
semilogy(1:it_ab, err_ab, '--', 'LineWidth', lw, 'DisplayName', 'non-hybrid AB'); hold on;
semilogy(1:it_ba, err_ba, ':', 'LineWidth', lw, 'DisplayName', 'non-hybrid BA');
semilogy(1:it_hab, err_hab, '-', 'LineWidth', lw, 'DisplayName', 'hybrid AB');
semilogy(1:it_hba, err_hba, '-.', 'LineWidth', lw, 'DisplayName', 'hybrid BA');
hold off;
title('Relative Error vs. Iteration');
xlabel('Iteration k');
ylabel('||x_k - x_{true}|| / ||x_{true}||');
legend('Location', 'Best');
grid on;
ylim([1e-4, 2]); % Adjust ylim as needed

% Subplot for Relative Residual
subplot(1, 2, 2);
semilogy(1:it_ab, res_ab, '--', 'LineWidth', lw, 'DisplayName', 'non-hybrid AB'); hold on;
semilogy(1:it_ba, res_ba, ':', 'LineWidth', lw, 'DisplayName', 'non-hybrid BA');
semilogy(1:it_hab, res_hab, '-', 'LineWidth', lw, 'DisplayName', 'hybrid AB');
semilogy(1:it_hba, res_hba, '-.', 'LineWidth', lw, 'DisplayName', 'hybrid BA');
hold off;
title('Relative Residual vs. Iteration');
xlabel('Iteration k');
ylabel('||b - Ax_k|| / ||b||');
legend('Location', 'Best');
grid on;
ylim([1e-7, 2]); % Adjust ylim as needed

%% Plot 6: Visual Comparison of Final Solutions
fprintf('Generating final solution comparison plot...\n');

figure('Name', 'Final Solution Comparison', 'Position', [100 100 900 700]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Comparison of Final Computed Solutions vs. True Solution', 'FontSize', 14);

% Non-hybrid AB
nexttile;
plot(x_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution'); hold on;
plot(x_ab, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Computed (AB)');
hold off; grid on; title('non-hybrid AB-GMRES');
xlabel('Element index'); ylabel('Value'); legend('Location', 'Best');

% Non-hybrid BA
nexttile;
plot(x_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution'); hold on;
plot(x_ba, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Computed (BA)');
hold off; grid on; title('non-hybrid BA-GMRES');
xlabel('Element index'); legend('Location', 'Best');

% Hybrid AB
nexttile;
plot(x_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution'); hold on;
plot(x_hab, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Computed (Hybrid AB)');
hold off; grid on; title('hybrid AB-GMRES');
xlabel('Element index'); ylabel('Value'); legend('Location', 'Best');

% Hybrid BA
nexttile;
plot(x_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution'); hold on;
plot(x_hba, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Computed (Hybrid BA)');
hold off; grid on; title('hybrid BA-GMRES');
xlabel('Element index'); legend('Location', 'Best');

fprintf('All plotting complete.\n');