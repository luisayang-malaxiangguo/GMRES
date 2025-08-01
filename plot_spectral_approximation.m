function plot_spectral_approximation()
% PLOT_SPECTRAL_APPROXIMATION
% Visualizes how well the harmonic Ritz values (Theta) approximate the
% squared singular values (mu) of the matrix A for each GMRES variant.

%% 1) Set up Test Problem & Parameters
n = 32;
[A, b, x_true] = deriv2(n); % 'deriv2' is a good choice here
tol = 1e-8;
maxit = n;
lambda = 1e-3;
B = A';
rng(0);
DeltaM = 1e-5 * randn(size(A'));

% Get the "true" singular values (squared) of A
[~, S_true, ~] = svd(A, 'econ');
mu_true = diag(S_true).^2;
mu_true = sort(mu_true, 'ascend'); % Sort for easier comparison

%% 2) Run methods to get iterative history
fprintf('Running GMRES variants to collect spectral data...\n');
[~, ~, ~, it_ab, ~, ~, phi_ab_iter, ~] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
[~, ~, ~, it_ba, ~, ~, phi_ba_iter, ~] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
[~, ~, ~, it_hab, ~, ~, phi_hab_iter, ~] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
[~, ~, ~, it_hba, ~, ~, phi_hba_iter, ~] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
fprintf('Simulations complete.\n');

%% 3) Re-run to extract Theta values at each iteration
% We need to modify the original functions slightly to output Theta_iter
% For this script, we'll just re-implement the core logic to get Theta.
k_values = [5, 15, 30]; % Iterations to inspect

figure('Name', 'Spectral Approximation Quality', 'Position', [100 100 900 700]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Harmonic Ritz Values (\theta) vs. True Singular Values (\mu)', 'FontSize', 14);

% --- Plot for non-hybrid AB ---
ax1 = nexttile;
plot_theta_vs_mu(ax1, 'ab_nonhybrid', A, B, b, maxit, mu_true, k_values, lambda);
title('non-hybrid AB-GMRES');
ylabel('Value (log scale)');

% --- Plot for non-hybrid BA ---
ax2 = nexttile;
plot_theta_vs_mu(ax2, 'ba_nonhybrid', A, B, b, maxit, mu_true, k_values, lambda);
title('non-hybrid BA-GMRES');

% --- Plot for hybrid AB ---
ax3 = nexttile;
plot_theta_vs_mu(ax3, 'ab_hybrid', A, B, b, maxit, mu_true, k_values, lambda);
title('hybrid AB-GMRES');
xlabel('Index'); ylabel('Value (log scale)');

% --- Plot for hybrid BA ---
ax4 = nexttile;
plot_theta_vs_mu(ax4, 'ba_hybrid', A, B, b, maxit, mu_true, k_values, lambda);
title('hybrid BA-GMRES');
xlabel('Index');

end

function plot_theta_vs_mu(ax, method, A, B, b, maxit, mu_true, k_vals, lambda)
    hold(ax, 'on');
    colors = lines(length(k_vals));

    % Plot true singular values as a reference line
    semilogy(ax, 1:length(mu_true), mu_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \mu_i');

    for i = 1:length(k_vals)
        k = k_vals(i);
        Theta_k = get_harmonic_ritz_values(method, A, B, b, k, lambda);
        semilogy(ax, 1:k, Theta_k, 'o', 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), 'DisplayName', sprintf('k = %d', k));
    end

    hold(ax, 'off');
    grid on;
    legend('Location', 'SouthEast');
    ylim([1e-6, 1e4]); % Adjust as needed
end

function Theta = get_harmonic_ritz_values(method, A, B, b, k_target, lambda)
    % Simplified runner to extract harmonic Ritz values at iteration k_target
    maxit = k_target;
    
    % Setup based on method
    if contains(method, 'ab')
        r0 = b;
        m_space = size(A,1);
        Q = zeros(m_space, maxit + 1);
        H = zeros(maxit + 1, maxit);
        beta = norm(r0);
        Q(:,1) = r0 / beta;
    else % ba
        r0 = B*b;
        n_space = size(A,2);
        Q = zeros(n_space, maxit + 1);
        H = zeros(maxit + 1, maxit);
        beta = norm(r0);
        Q(:,1) = r0 / beta;
    end
    
    % Arnoldi process
    for k = 1:k_target
        if contains(method, 'ab')
            v = A * (B * Q(:,k));
        else % ba
            v = B * (A * Q(:,k));
        end
        for j = 1:k
            H(j,k) = Q(:,j)'*v;
            v = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0, break; end
        Q(:,k+1) = v / H(k+1,k);
    end

    % Calculate harmonic Ritz values at k_target
    Hk_small = H(1:k_target, 1:k_target);
    ek = zeros(k_target,1);
    ek(end) = 1;
    
    if contains(method, 'nonhybrid')
        P_eig_problem = Hk_small + (H(k_target+1, k_target)^2) * (Hk_small'\(ek*ek'));
        [~, D_eig] = eig(P_eig_problem);
        Theta = sort(real(diag(D_eig)), 'ascend');
    else % hybrid
         P_unreg = Hk_small + (H(k_target+1, k_target)^2) * (Hk_small'\(ek*ek'));
         P_reg = P_unreg + lambda*eye(k_target);
         [~, D_eig] = eig(P_reg);
         Theta = sort(real(diag(D_eig)), 'ascend');
    end
end