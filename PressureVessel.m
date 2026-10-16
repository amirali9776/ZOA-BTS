clc; clear;

% ----------------------------------------

% Pressure Vessel Design (CONTINUOUS)

% x = [Ts, Th, R, L]

% Ts: shell thickness

% Th: head thickness

% R : inner radius

% L : length (cylindrical section)

% ----------------------------------------

dim = 4;

% کلاسیک (continuous bounds)

lb = [0.0625, 0.0625, 10, 10];

ub = [10,     10,     200, 240];

fe_limit = 200000;

popSize  = 1000;

func = @(x) pv_penalty(x);

% ========== Multi-run settings ==========

N_RUNS = 20;

base_seed = 1;

algs = { ...

    struct('name','ZOA',  'run',@(dim,lb,ub,fe,pop,fun) ZOA_Budgeted(dim,lb,ub,fe,pop,fun)); ...

};

nAlg = numel(algs);

best_fit_runs = inf(N_RUNS, nAlg);

true_fit_runs = inf(N_RUNS, nAlg);

maxviol_runs  = zeros(N_RUNS, nAlg);

FE_runs       = zeros(N_RUNS, nAlg);

best_x_runs   = nan(N_RUNS, dim, nAlg);

overall_best_pen  = inf(1, nAlg);

overall_best_x    = nan(nAlg, dim);

overall_best_true = inf(1, nAlg);

overall_best_viol = inf(1, nAlg);

for r = 1:N_RUNS

    for a = 1:nAlg

        rng(base_seed + r, "twister");

        [best_x, best_fit, curve, FE] = algs{a}.run(dim, lb, ub, fe_limit, popSize, func);

        [f_true, g] = pv_true(best_x);

        max_viol = max(g);

        best_fit_runs(r,a) = best_fit;

        true_fit_runs(r,a) = f_true;

        maxviol_runs(r,a)  = max_viol;

        best_x_runs(r,:,a) = best_x;

        FE_runs(r,a)       = FE;

        if best_fit < overall_best_pen(a)

            overall_best_pen(a)  = best_fit;

            overall_best_x(a,:)  = best_x;

            overall_best_true(a) = f_true;

            overall_best_viol(a) = max_viol;

        end

        fprintf("[%s] Run %02d/%02d | Pen=%.6g | True=%.6g | MaxViol=%.3g | FE=%d\n", ...

            algs{a}.name, r, N_RUNS, best_fit, f_true, max_viol, FE);

    end

end

% ===================== SUMMARY =====================

fprintf("\n================ SUMMARY (%d runs) ================\n", N_RUNS);

for a = 1:nAlg

    fprintf("\n----- %s -----\n", algs{a}.name);

    fprintf("Penalized : Best=%.10g | Mean=%.10g | Std=%.10g\n", ...

        min(best_fit_runs(:,a)), mean(best_fit_runs(:,a)), std(best_fit_runs(:,a)));

    fprintf("True cost : Best=%.10g | Mean=%.10g | Std=%.10g\n", ...

        min(true_fit_runs(:,a)), mean(true_fit_runs(:,a)), std(true_fit_runs(:,a)));

    fprintf("MaxViol   : Best=%.6g | Mean=%.6g\n", ...

        min(maxviol_runs(:,a)), mean(maxviol_runs(:,a)));

    fprintf("Overall Best x = [%.6f %.6f %.6f %.6f] | Best Pen=%.10g | True=%.10g | MaxViol=%.3g\n", ...

        overall_best_x(a,1), overall_best_x(a,2), overall_best_x(a,3), overall_best_x(a,4), ...

        overall_best_pen(a), overall_best_true(a), overall_best_viol(a));

end

% ===================== TABLE =====================

T = make_results_table(algs, best_fit_runs, true_fit_runs, best_x_runs, maxviol_runs);

disp(T);

% Optional save:

% writetable(T, 'PV_Results.csv');

function T = make_results_table(algs, best_fit_runs, true_fit_runs, best_x_runs, maxviol_runs)

    nAlg = numel(algs);

    dim  = size(best_x_runs, 2);

    N_RUNS = size(best_fit_runs, 1);

    AlgName = strings(nAlg,1);

    Best_f  = zeros(nAlg,1);

    Mean_f  = zeros(nAlg,1);

    Std_f   = zeros(nAlg,1);

    Best_pen = zeros(nAlg,1);

    Mean_pen = zeros(nAlg,1);

    Std_pen  = zeros(nAlg,1);

    Best_x  = zeros(nAlg, dim);

    Mean_x  = zeros(nAlg, dim);

    Std_x   = zeros(nAlg, dim);

    Best_MaxViol = zeros(nAlg,1);

    Mean_MaxViol = zeros(nAlg,1);

    for a = 1:nAlg

        AlgName(a) = string(algs{a}.name);

        % ---- Penalized statistics ----

        pen_runs = best_fit_runs(:,a);

        Mean_pen(a) = mean(pen_runs);

        Std_pen(a)  = std(pen_runs);

        [Best_pen(a), bestRunIdx] = min(pen_runs);   % انتخاب بر اساس penalized

        % ---- True objective (corresponding to best penalized) ----

        f_runs = true_fit_runs(:,a);

        Mean_f(a) = mean(f_runs);

        Std_f(a)  = std(f_runs);

        Best_f(a) = f_runs(bestRunIdx);

        % ---- x statistics ----

        X_runs = squeeze(best_x_runs(:,:,a));   % N_RUNS x dim

        Mean_x(a,:) = mean(X_runs, 1);

        Std_x(a,:)  = std(X_runs, 0, 1);

        Best_x(a,:) = X_runs(bestRunIdx, :);

        % ---- violation ----

        Best_MaxViol(a) = min(maxviol_runs(:,a));

        Mean_MaxViol(a) = mean(maxviol_runs(:,a));

    end

    % ---- base table ----

    T = table(AlgName, Best_pen, Mean_pen, Std_pen, ...

              Best_f, Mean_f, Std_f, ...

              Best_MaxViol, Mean_MaxViol, ...

              'VariableNames', { ...

                  'Alg', ...

                  'Best_pen','Mean_pen','Std_pen', ...

                  'Best_f','Mean_f','Std_f', ...

                  'Best_MaxViol','Mean_MaxViol'});

    % ---- add x columns dynamically (works for any dim) ----

    for j = 1:dim

        T = addvars(T, ...

            Best_x(:,j), Mean_x(:,j), Std_x(:,j), ...

            'Before','Best_MaxViol', ...

            'NewVariableNames', { ...

                sprintf('Best_x%d',j), ...

                sprintf('Mean_x%d',j), ...

                sprintf('Std_x%d',j)});

    end

end

% ===================== Functions =====================

function f_pen = pv_penalty(x)

    [f, g] = pv_true(x);

    viol = max(0, g);

    V = sum(viol);

    % Adaptive rho (مثل خودت)

    if V > 1e-2

        rho = 1e9;

    elseif V > 1e-4

        rho = 1e8;

    elseif V > 1e-6

        rho = 1e7;

    else

        rho = 1e6;

    end

    f_pen = f + rho * sum(viol.^2);

end

function [f, g] = pv_true(x)

    % x = [Ts, Th, R, L]

    Ts = x(1);  Th = x(2);  R = x(3);  L = x(4);

    % ---- Objective (classic pressure vessel cost) ----

    % f = 0.6224*Ts*R*L + 1.7781*Th*R^2 + 3.1661*Ts^2*L + 19.84*Ts^2*R

    f = 0.6224*Ts*R*L + 1.7781*Th*R^2 + 3.1661*(Ts^2)*L + 19.84*(Ts^2)*R;

    % ---- Constraints (g<=0 feasible) ----

    % g1: Ts >= 0.0193*R

    g1 = 0.0193*R - Ts;

    % g2: Th >= 0.00954*R

    g2 = 0.00954*R - Th;

    % g3: Volume >= 1296000  (π R^2 L + 4/3 π R^3 >= 1296000)

    g3 = 1296000 - (pi*R^2*L + (4/3)*pi*R^3);

    % g4: L <= 240  (یا L - 240 <= 0)

    g4 = L - 240;

    g = [g1 g2 g3 g4];

end

% =========================================================================

% ZOA (BUDGETED)  --- Robust + Fair + Reflection Bounds

% FIXED: guided / DM / SL directions

% =========================================================================

function [zahhak_pos, zahhak_fit, convergence_curve, FE] = ...

    ZOA_Budgeted(dim, lb, ub, fe_limit, popSize, func)

FE = 0;

convergence_curve = inf(1, fe_limit);

%% ---- Bounds handling (scalar or vector) ----

if isscalar(lb)

    lb_vec = ones(1, dim) * lb;

    ub_vec = ones(1, dim) * ub;

else

    lb_vec = lb(:)';    % 1 x dim

    ub_vec = ub(:)';    % 1 x dim

end

%% --- Initialization with Logistic Map ---

mu = 4.0;

chaos_matrix = rand(popSize, dim);

for k = 1:10

    chaos_matrix = mu * chaos_matrix .* (1 - chaos_matrix);

end

if isscalar(lb)

    soldiers = lb + chaos_matrix * (ub - lb);

else

    soldiers = repmat(lb_vec, popSize, 1) + ...

               chaos_matrix .* repmat(ub_vec - lb_vec, popSize, 1);

end

% Robust bounds repair (Reflection)

soldiers = BoundReflect(soldiers, lb_vec, ub_vec);

fitnesses = inf(popSize, 1);

%% --- Initial Evaluation ---

best_so_far = inf;

for i = 1:popSize

    if FE >= fe_limit, break; end

    fitnesses(i) = func(soldiers(i, :));

    FE = FE + 1;

    best_so_far = min(best_so_far, fitnesses(i));

    convergence_curve(FE) = best_so_far;

end

%% --- Find Zahhak (Global Best) ---

[zahhak_fit, min_idx] = min(fitnesses);

zahhak_pos = soldiers(min_idx, :);

indices = 1:popSize;

%% ================= Main Loop =================

while FE < fe_limit

    %% ---- Khayyam coefficient ----

    lambda_t = FE / fe_limit;

    alpha = 0.5 + 1.5 * lambda_t;

    r3 = rand();

    r4 = rand();

    cubic_decay = (1 + lambda_t^3);

    % --- Stabilized gamma (bounded) ---

    gamma_raw = alpha * cubic_decay * sin(2*pi*r3) * cos(2*pi*r4);

    gamma_khayyam = tanh(gamma_raw);   % stays in [-1,1]

    %% ---- Soldier movement (exclude Zahhak index) ----

    mask = indices ~= min_idx;

    active_indices = indices(mask);

    n_active = numel(active_indices);

    if n_active <= 0

        break;

    end

    % Save old positions/fitness for Greedy selection

    old_pos = soldiers(active_indices, :);

    old_fit = fitnesses(active_indices);

    r_vec3 = rand(n_active, dim); 

    partner_idx = randi(popSize, n_active, 1);

    % -------------------------------------------------

    % FIX #1: guided must be gamma * (best - old)

    % -------------------------------------------------

    guided = gamma_khayyam * r_vec3 .* (repmat(zahhak_pos, n_active, 1) - soldiers(partner_idx, :));

    % Stochastic decision mechanism: DM vs SL

    p = rand(n_active, 1);                 % switch probability

    r_vec = rand(n_active, dim);           % random scaling in [0,1]

    theta = 2*pi*rand(n_active, dim);      % random angles

    % -------------------------------------------------

    % FIX #2: DM should be differential (best - old), not absolute best

    % -------------------------------------------------

    DM = r_vec .* sin(theta) .* (repmat(zahhak_pos, n_active, 1) - old_pos);

    % -------------------------------------------------

    % FIX #3: Social Learning direction should be (partner - old)

    % partner sampled from FULL population, partner ~= self

    % -------------------------------------------------

    partner_idx = randi(popSize, n_active, 1);

    self_idx = active_indices(:);

    same = (partner_idx == self_idx);

    while any(same)

        partner_idx(same) = randi(popSize, sum(same), 1);

        same = (partner_idx == self_idx);

    end

    r_vec2 = rand(n_active, dim);

    SL = r_vec2 .* (soldiers(partner_idx, :) - old_pos);

    % Choose between DM and SL

    A = zeros(n_active, dim);

    use_DM = (p > 0.5);

    A(use_DM, :)  = DM(use_DM, :);

    A(~use_DM, :) = SL(~use_DM, :);

    % Candidate new positions

    new_pos = old_pos + guided + A;

    % Robust bounds repair (Reflection instead of clamp)

    new_pos = BoundReflect(new_pos, lb_vec, ub_vec);

    % Evaluate and Greedy select

    for k = 1:n_active

        if FE >= fe_limit, break; end

        cand = new_pos(k, :);

        f_cand = func(cand);

        FE = FE + 1;

        if f_cand < old_fit(k)

            % accept

            idxk = active_indices(k);

            soldiers(idxk, :) = cand;

            fitnesses(idxk)   = f_cand;

            if f_cand < zahhak_fit

                zahhak_fit = f_cand;

                zahhak_pos = cand;

                min_idx = idxk;

            end

        end

        convergence_curve(FE) = zahhak_fit;

    end

    if FE >= fe_limit

        break;

    end

    %% ================= BTS Local Search (as you had, with Reflection repair) =================

    if isscalar(lb)

        Delta = (ub - lb);

    else

        Delta = mean(ub_vec - lb_vec);

    end

    omega = 0.1;

    progress = FE / fe_limit;

    radius = (Delta * omega * (1 - progress)^3) + 1e-10;

    base_pos = zahhak_pos;

    base_fit = zahhak_fit;

    % Choose 2 dimensions (or 1 if dim==1)

    if dim == 1

        dims_selected = 1;

        k_dims = 1;

    else

        dims_perm = randperm(dim);

        dims_selected = dims_perm(1:2);

        k_dims = 2;

    end

    if k_dims == 2

        % Gram–Schmidt for *2D* basis

        r1 = randn(2,1);

        r2 = randn(2,1);

        u = r1 / (norm(r1) + 1e-12);

        v = r2 - (dot(r2,u) * u);

        v = v / (norm(v) + 1e-12);

        phi = 2*pi*rand();

        hex_angles = (0:5) * (pi/3);

        best_hex_fit = base_fit;

        best_hex_pos = base_pos;

        for a = hex_angles

            if FE >= fe_limit, break; end

            ang = a + phi;

            offset2 = radius * (cos(ang) * u + sin(ang) * v); % *2x1*

            cand = base_pos;

            cand(dims_selected(1)) = cand(dims_selected(1)) + offset2(1);

            cand(dims_selected(2)) = cand(dims_selected(2)) + offset2(2);

            cand = BoundReflect(cand, lb_vec, ub_vec);

            f_cand = func(cand);

            FE = FE + 1;

            if f_cand < best_hex_fit

                best_hex_fit = f_cand;

                best_hex_pos = cand;

            end

            convergence_curve(FE) = min(zahhak_fit, best_hex_fit);

        end

        if best_hex_fit < base_fit

            base_fit = best_hex_fit;

            base_pos = best_hex_pos;

            zahhak_fit = base_fit;

            zahhak_pos = base_pos;

            soldiers(min_idx, :) = zahhak_pos;

            fitnesses(min_idx)   = zahhak_fit;

            if FE <= fe_limit

                convergence_curve(FE) = zahhak_fit;

            end

        else

            % Fallback axis steps

            best_ax_fit = base_fit;

            best_ax_pos = base_pos;

            for d = dims_selected

                for s = [-1, 1]

                    if FE >= fe_limit, break; end

                    rho = rand();

                    cand = base_pos;

                    cand(d) = cand(d) + s * rho * radius;

                    cand = BoundReflect(cand, lb_vec, ub_vec);

                    f_cand = func(cand);

                    FE = FE + 1;

                    if f_cand < best_ax_fit

                        best_ax_fit = f_cand;

                        best_ax_pos = cand;

                    end

                    convergence_curve(FE) = min(zahhak_fit, best_ax_fit);

                end

                if FE >= fe_limit, break; end

            end

            if best_ax_fit < base_fit

                base_fit = best_ax_fit;

                base_pos = best_ax_pos;

                zahhak_fit = base_fit;

                zahhak_pos = base_pos;

                soldiers(min_idx, :) = zahhak_pos;

                fitnesses(min_idx)   = zahhak_fit;

                convergence_curve(FE) = zahhak_fit;

            end

        end

    else

        % dim==1 local

        best_1d_fit = base_fit;

        best_1d_pos = base_pos;

        for s = [-1, 1]

            if FE >= fe_limit, break; end

            rho = rand();

            cand = base_pos;

            cand(1) = cand(1) + s * rho * radius;

            cand = BoundReflect(cand, lb_vec, ub_vec);

            f_cand = func(cand);

            FE = FE + 1;

            if f_cand < best_1d_fit

                best_1d_fit = f_cand;

                best_1d_pos = cand;

            end

            convergence_curve(FE) = min(zahhak_fit, best_1d_fit);

        end

        if best_1d_fit < base_fit

            zahhak_fit = best_1d_fit;

            zahhak_pos = best_1d_pos;

            soldiers(min_idx, :) = zahhak_pos;

            fitnesses(min_idx)   = zahhak_fit;

            convergence_curve(FE) = zahhak_fit;

        end

    end

end

convergence_curve = convergence_curve(1:FE);

end

% =========================================================================

% Robust bound repair: Reflection + folding (vector-safe)

% =========================================================================

function X = BoundReflect(X, lb_vec, ub_vec)

    lb_vec = lb_vec(:)';

    ub_vec = ub_vec(:)';

    n = size(X,1);

    LB = repmat(lb_vec, n, 1);

    UB = repmat(ub_vec, n, 1);

    % reflect once

    idxL = X < LB;

    X(idxL) = 2*LB(idxL) - X(idxL);

    idxU = X > UB;

    X(idxU) = 2*UB(idxU) - X(idxU);

    % fold extreme overshoots robustly

    range = (UB - LB) + eps;

    X = LB + mod(X - LB, 2*range);

    idxU2 = X > UB;

    X(idxU2) = UB(idxU2) - (X(idxU2) - UB(idxU2));

    % final safety clamp (rare)

    X = max(min(X, UB), LB);

end
