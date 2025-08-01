function plot_perturbation_bound_validation()
% PLOT_PERTURBATION_BOUND_VALIDATION Validates the theoretical perturbation bounds
% by comparing them against the actual change in filter factors.

%% 1) Set up Test Problem & Parameters
n = 32;
[A, b, x_true] = shaw(n); % Use 'shaw' or 'deriv2'
tol = 1e-8;
maxit = n;
lambda = 1e-3;
B = A';
rng(0);
pert_magnitude = 1e-5;
DeltaM_pert = pert_magnitude * randn(size(A'));
DeltaM_zero = zeros(size(A'));

%% 2) Run simulations to get perturbed and unperturbed factors
fprintf('Running simulations for bound validation...\n');

% Run WITH perturbation
[~, ~, ~, it_ab_p, ~, dphi_ab_p, phi_ab_p, ~] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_pert);
[~, ~, ~, it_ba_p, ~, dphi_ba_p, phi_ba_p, ~] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_pert);
[~, ~, ~, it_hab_p, ~, dphi_hab_p, phi_hab_p, ~] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_pert);
[~, ~, ~, it_hba_p, ~, dphi_hba_p, phi_hba_p, ~] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_pert);

% Run WITHOUT perturbation (DeltaM = 0)
[~, ~, ~, ~, ~, ~, phi_ab_u, ~] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_zero);
[~, ~, ~, ~, ~, ~, phi_ba_u, ~] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_zero);
[~, ~, ~, ~, ~, ~, phi_hab_u, ~] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_zero);
[~, ~, ~, ~, ~, ~, phi_hba_u, ~] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_zero);

fprintf('Simulations complete.\n');

%% 2.5) Determine k_to_plot (clamped to available iterations)
desired_k = 20;
all_lengths = [ numel(phi_ab_p), numel(phi_ab_u), ...
                numel(phi_ba_p), numel(phi_ba_u), ...
                numel(phi_hab_p), numel(phi_hab_u), ...
                numel(phi_hba_p), numel(phi_hba_u) ];
max_k = min(all_lengths);               % smallest available length across all methods
k_to_plot = min(desired_k, max_k);      % clamp to available range
if desired_k > max_k
    warning('Requested k = %d exceeds available iterations (%d). Using k = %d instead.', ...
            desired_k, max_k, k_to_plot);
end

%% 3) Plot the comparison
fprintf('Generating bound validation plot for k = %d...\n', k_to_plot);

figure('Name', 'Perturbation Bound Validation', 'Position', [100 100 900 700]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, ['Validation of Perturbation Bounds at Iteration k = ', num2str(k_to_plot)], 'FontSize', 14);

% non-hybrid AB
phi_ab_p_k = phi_ab_p{k_to_plot};
phi_ab_u_k = phi_ab_u{k_to_plot};
actual_change_ab     = abs(phi_ab_p_k(1:k_to_plot) - phi_ab_u_k(1:k_to_plot));
theoretical_bound_ab = abs(dphi_ab_p(1:k_to_plot));
semilogy(1:k_to_plot, actual_change_ab,     'o-','DisplayName','Actual Change |Δφ|'); hold on;
semilogy(1:k_to_plot, theoretical_bound_ab, 'x--','DisplayName','Theoretical Bound |δφ|');
hold off; grid on; title('non-hybrid AB-GMRES');
xlabel('Mode index i'); ylabel('Magnitude'); legend('Location','Best');

% --- non-hybrid BA ---
phi_ba_p_k = phi_ba_p{k_to_plot};
phi_ba_u_k = phi_ba_u{k_to_plot};
actual_change_ba     = abs(phi_ba_p_k(1:k_to_plot) - phi_ba_u_k(1:k_to_plot));
theoretical_bound_ba = abs(dphi_ba_p(1:k_to_plot));
semilogy(1:k_to_plot, actual_change_ba,     'o-','DisplayName','Actual Change |Δφ|'); hold on;
semilogy(1:k_to_plot, theoretical_bound_ba, 'x--','DisplayName','Theoretical Bound |δφ|');
hold off; grid on; title('non-hybrid BA-GMRES');
xlabel('Mode index i'); legend('Location','Best');

% --- hybrid AB ---
phi_hab_p_k = phi_hab_p{k_to_plot};
phi_hab_u_k = phi_hab_u{k_to_plot};
actual_change_hab     = abs(phi_hab_p_k(1:k_to_plot) - phi_hab_u_k(1:k_to_plot));
theoretical_bound_hab = abs(dphi_hab_p(1:k_to_plot));
semilogy(1:k_to_plot, actual_change_hab,     'o-','DisplayName','Actual Change |Δφ|'); hold on;
semilogy(1:k_to_plot, theoretical_bound_hab, 'x--','DisplayName','Theoretical Bound |δφ|');
hold off; grid on; title('hybrid AB-GMRES');
xlabel('Mode index i'); ylabel('Magnitude'); legend('Location','Best');

% --- hybrid BA ---
phi_hba_p_k = phi_hba_p{k_to_plot};
phi_hba_u_k = phi_hba_u{k_to_plot};
actual_change_hba     = abs(phi_hba_p_k(1:k_to_plot) - phi_hba_u_k(1:k_to_plot));
theoretical_bound_hba = abs(dphi_hba_p(1:k_to_plot));
semilogy(1:k_to_plot, actual_change_hba,     'o-','DisplayName','Actual Change |Δφ|'); hold on;
semilogy(1:k_to_plot, theoretical_bound_hba, 'x--','DisplayName','Theoretical Bound |δφ|');
hold off; grid on; title('hybrid BA-GMRES');
xlabel('Mode index i'); legend('Location','Best');

end