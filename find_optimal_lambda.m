function find_optimal_lambda()
% FIND_OPTIMAL_LAMBDA Finds the optimal Tikhonov regularization parameter
% for hybrid GMRES methods, or solves for non-hybrid methods.
%
%
% For HYBRID methods:
%   It uses MATLAB's 'fminbnd' to find the value of lambda that minimizes
%   the GCV function.
%
% For NON-HYBRID methods:
%   It skips the lambda optimization and directly solves the system.

    % --- 1. CHOOSE METHOD AND TEST PROBLEM ---
    
    % ---CHOOSE WHICH METHOD TO RUN ---
    % Options: 'hybrid_ab', 'hybrid_ba', 'nonhybrid_ab', 'nonhybrid_ba'
    method_to_run = 'hybrid_ab'; 
    
    n = 64; % Problem size
    problem_to_run = 'shaw';
    fprintf('Setting up test problem: %s, with n = %d.\n', problem_to_run, n);
    [A, b, x_true] = generate_test_problem(problem_to_run, n);
    m = size(A, 1);
    
    % Define a perturbation for the B matrix
    rng(42);
    E = 1e-4 * randn(size(A'));
    B = A' + E;
    
    % The perturbation DeltaM depends on the method (AB vs BA)
    DeltaM_AB = A * E;
    DeltaM_BA = E * A;

    fprintf('Selected method: %s\n\n', strrep(method_to_run, '_', '-'));
    
    % --- 2.SOLVE BASED ON METHOD ---
    
    % Use a switch statement to handle the different methods
    switch method_to_run
        case {'hybrid_ab', 'hybrid_ba'}
            % --- This block runs for HYBRID methods ---
            
            % Find optimal lambda using the optimizer
            lambda_interval = [1e-9, 1e-1]; 
            fprintf('Searching for optimal lambda in the interval [%.2e, %.2e]...\n', lambda_interval(1), lambda_interval(2));
            
            k_gcv = 20;
            % The GCV function needs to know if it's 'ab' or 'ba'
            gcv_type = extractAfter(method_to_run, 'hybrid_');
            gcv_handle = @(lambda) gcv_function(lambda, A, B, b, m, k_gcv, gcv_type);
            
            options = optimset('Display', 'iter', 'TolX', 1e-8);
            [lambda_optimal, gcv_min] = fminbnd(gcv_handle, lambda_interval(1), lambda_interval(2), options);
            
            fprintf('\nOptimization complete.\n');
            fprintf('Optimal lambda found: %e\n', lambda_optimal);
            
            % Solve the system with the optimal lambda
            fprintf('\nSolving the system with the optimal lambda...\n');
            maxit = 32;
            tol = 1e-6;
            if strcmp(method_to_run, 'hybrid_ab')
                [x_final, err, res, niters] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda_optimal, DeltaM_AB);
            else
                [x_final, err, res, niters] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda_optimal, DeltaM_BA);
            end
            
            % Plot GCV Curve
            plot_gcv_curve(gcv_handle, lambda_interval, lambda_optimal, gcv_min);

        case {'nonhybrid_ab', 'nonhybrid_ba'}
            % --- runs for NON-HYBRID methods ---
            
            fprintf('Non-hybrid method selected. Skipping lambda optimization.\n');
            lambda_optimal = 0; % Lambda is not used
            
            % Solve the system directly
            maxit = 32;
            tol = 1e-6;
            if strcmp(method_to_run, 'nonhybrid_ab')
                [x_final, err, res, niters] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_AB);
            else
                [x_final, err, res, niters] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM_BA);
            end

        otherwise
            error('Unknown method specified in "method_to_run".');
    end
    
    % --- 3. DISPLAY RESULTS AND PLOT SOLUTION ---
    fprintf('\n--- Final Results for %s ---\n', strrep(method_to_run, '_', '-'));
    fprintf('Iterations performed: %d\n', niters);
    fprintf('Final relative error norm: %.4f\n', err(end));
    fprintf('Final relative residual norm: %.4f\n', res(end));
    
    % Plot the final solution
    plot_solution(x_true, x_final, n, method_to_run, lambda_optimal);
    
end

% =========================================================================
% Nested GCV Function - handles both AB and BA GMRES
% =========================================================================
function gcv_val = gcv_function(lambda, A, B, b, m, k_gcv, gcv_type)
    
    % The initial residual and Arnoldi step depend on the method type.
    if strcmp(gcv_type, 'ab')
        r0 = b;
        n_arnoldi = m; % Arnoldi is in m-space
    else % 'ba'
        r0 = B * b;
        n_arnoldi = size(A, 2); % Arnoldi is in n-space
    end
    
    beta = norm(r0);
    Q = zeros(n_arnoldi, k_gcv + 1);
    H = zeros(k_gcv + 1, k_gcv);
    Q(:,1) = r0 / beta;
    e1 = [beta; zeros(k_gcv, 1)];

    for k = 1:k_gcv
        if strcmp(gcv_type, 'ab')
            v = A * (B * Q(:,k));
        else % 'ba'
            v = B * (A * Q(:,k));
        end
        
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12, break; end
        Q(:,k+1) = v / H(k+1,k);
    end
    k = size(H, 2);
    
    Hk = H(1:k+1, 1:k);
    tk = e1(1:k+1);
    
    yk = (Hk' * Hk + lambda * eye(k)) \ (Hk' * tk);
    
    residual_norm_sq = norm(tk - Hk * yk)^2;
    
    [~, S, ~] = svd(H(1:k, 1:k), 'econ');
    s_diag = diag(S);
    
    % trace term depends on the method
    if strcmp(gcv_type, 'ab')
        trace_m = m;
    else % 'ba'
        trace_m = size(A,2);
    end
    trace_val = sum(s_diag.^2 ./ (s_diag.^2 + lambda));
    denominator = (trace_m - trace_val)^2;
    
    gcv_val = residual_norm_sq / denominator;
    
    if isnan(gcv_val) || isinf(gcv_val) || denominator < eps
        gcv_val = 1e20;
    end
end

% =========================================================================
% Plotting and Problem Generation Functions
% =========================================================================
function plot_gcv_curve(gcv_handle, interval, lambda_opt, gcv_min)
    figure;
    lambdas_for_plot = logspace(log10(interval(1)), log10(interval(2)), 100);
    gcv_values = zeros(size(lambdas_for_plot));
    for i = 1:length(lambdas_for_plot)
        gcv_values(i) = gcv_handle(lambdas_for_plot(i));
    end
    loglog(lambdas_for_plot, gcv_values, '-b', 'LineWidth', 2);
    hold on;
    loglog(lambda_opt, gcv_min, 'pr', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
    title('GCV Function vs. Lambda');
    xlabel('Regularization Parameter, \lambda');
    ylabel('GCV Function Value');
    legend('GCV Curve', 'Optimal \lambda found by fminbnd');
    grid on;
    axis tight;
end

function plot_solution(x_true, x_final, n, method_name, lambda)
    figure;
    plot(1:n, x_true, 'k-', 'LineWidth', 2);
    hold on;
    plot(1:n, x_final, 'r--', 'LineWidth', 1.5);
    
    clean_name = strrep(method_name, '_', '-');
    if contains(method_name, 'hybrid')
        title_str = sprintf('Solution Comparison for %s (\\lambda=%.2e)', clean_name, lambda);
        legend_str = sprintf('Regularized Solution (\\lambda=%.2e)', lambda);
    else
        title_str = sprintf('Solution Comparison for %s', clean_name);
        legend_str = 'Non-Regularized Solution';
    end
    title(title_str);
    xlabel('Element index');
    ylabel('Value');
    legend('True Solution', legend_str);
    grid on;
    axis tight;
end

function [A, b, x] = generate_test_problem(name, n)
    switch lower(name)
        case 'shaw'
            t = ( (1:n) - 0.5 ) * pi / n;
            [T, S] = meshgrid(t);
            K = (cos(S) + cos(T)).^2 .* (sin(pi*(sin(S)+sin(T))) ./ (pi*(sin(S)+sin(T)))).^2;
            K(isinf(K)|isnan(K)) = 1;
            h = pi/n;
            A = h*K;
            x = 2*exp(-6*(t-0.8).^2)' + exp(-2*(t+0.5).^2)';
        case 'heat'
            t = (0:n-1)'/(n-1);
            kappa = 0.5;
            [T, S] = meshgrid(t);
            h = 1/n;
            A = h * (1/sqrt(4*pi*kappa)) * exp(-(S-T).^2 / (4*kappa));
            x = t;
            x(t > 0.5) = 1 - t(t > 0.5);
            x = x + 0.5;
        case 'deriv2'
            A = spdiags(ones(n,1)*[-1, 2, -1], -1:1, n, n);
            A = full(A);
            A(1,1:2) = [1 -1];
            A(n,n-1:n) = [-1 1];
            A = A * (n-1);
            t = (0:n-1)'/(n-1);
            x = t.^2 .* (1-t).^2 .* exp(2*t);
        otherwise
            error('Unknown problem name. Use shaw, heat, or deriv2.');
    end
    b = A*x;
    rng('default');
    e = randn(size(b));
    e = e/norm(e);
    eta = 1e-3 * norm(b);
    b = b + eta*e;
end
