% =========================================================================
% Zahhak Optimization Algorithm (ZOA) with Buzjani Tiling Strategy (BTS)
% Category: Ancient-Based Metaheuristic Algorithm
% 
% Description: 
% This function executes the ZOA framework under a strict Function 
% Evaluation (FE) budget. It balances chaotic global exploration with 
% high-precision geometric local refinement (BTS) while enforcing 
% robust reflection boundary constraints.
%
% Inputs:
%   dim       - Integer: Dimensionality of the problem
%   lb        - Scalar/Vector: Lower bounds of the search space
%   ub        - Scalar/Vector: Upper bounds of the search space
%   fe_limit  - Integer: Maximum allowable function evaluations
%   popSize   - Integer: Total number of candidate solutions in the population
%   func      - Function Handle: The objective/cost function to minimize
%
% Outputs:
%   zahhak_pos        - 1xD Vector: The global optimal solution found
%   zahhak_fit        - Scalar: The fitness value of the optimal solution
%   convergence_curve - 1xFE Vector: Convergence history of the global best
%   FE                - Integer: Total FEs consumed
% =========================================================================

function [zahhak_pos, zahhak_fit, convergence_curve, FE] = ...
    ZOA_Budgeted(dim, lb, ub, fe_limit, popSize, func)

    FE = 0;
    convergence_curve = inf(1, fe_limit);

    %% ---- Bounds Handling (Scalar to Vector mapping) ----
    if isscalar(lb)
        lb_vec = ones(1, dim) * lb;
        ub_vec = ones(1, dim) * ub;
    else
        lb_vec = lb(:)';    
        ub_vec = ub(:)';    
    end

    %% ---- Initialization with Chaotic Logistic Map ----
    mu = 4.0;
    chaos_matrix = rand(popSize, dim);
    for k = 1:10
        chaos_matrix = mu * chaos_matrix .* (1 - chaos_matrix);
    end
    
    if isscalar(lb)
        solutions = lb + chaos_matrix * (ub - lb);
    else
        solutions = repmat(lb_vec, popSize, 1) + ...
                   chaos_matrix .* repmat(ub_vec - lb_vec, popSize, 1);
    end
    
    % Robust bounds repair (Reflection)
    solutions = BoundReflect(solutions, lb_vec, ub_vec);
    fitnesses = inf(popSize, 1);

    %% ---- Initial Evaluation ----
    best_so_far = inf;
    for i = 1:popSize
        if FE >= fe_limit, break; end
        fitnesses(i) = func(solutions(i, :));
        FE = FE + 1;
        best_so_far = min(best_so_far, fitnesses(i));
        convergence_curve(FE) = best_so_far;
    end

    %% ---- Identify Zahhak (Global Best Solution) ----
    [zahhak_fit, min_idx] = min(fitnesses);
    zahhak_pos = solutions(min_idx, :);
    indices = 1:popSize;

    %% ================= MAIN OPTIMIZATION LOOP =================
    while FE < fe_limit
        
        %% ---- Khayyam Adaptive Coefficient ----
        lambda_t = FE / fe_limit;
        alpha = 0.5 + 1.5 * lambda_t;
        r3 = rand();
        r4 = rand();
        cubic_decay = (1 + lambda_t^3);
        
        % Stabilized dynamic bounded weight
        gamma_raw = alpha * cubic_decay * sin(2*pi*r3) * cos(2*pi*r4);
        gamma_khayyam = tanh(gamma_raw);   % Confined to [-1, 1]

        %% ---- Population Movement (Excluding Zahhak) ----
        mask = indices ~= min_idx;
        active_indices = indices(mask);
        n_active = numel(active_indices);
        
        if n_active <= 0
            break;
        end
        
        % Save prior state for Greedy Selection
        old_pos = solutions(active_indices, :);
        old_fit = fitnesses(active_indices);
        
        r_vec3 = rand(n_active, dim); 
        partner_idx = randi(popSize, n_active, 1);
        
        % Phase 1: Guided Movement Toward Zahhak
        guided = gamma_khayyam * r_vec3 .* (repmat(zahhak_pos, n_active, 1) - solutions(partner_idx, :));
        
        % Stochastic Decision Variables
        p = rand(n_active, 1);                 % Switching probability
        r_vec = rand(n_active, dim);           % Random scaling [0,1]
        theta = 2*pi*rand(n_active, dim);      % Random orientation angles
        
        % Phase 2: Deceptive Maneuver (DM)
        DM = r_vec .* sin(theta) .* (repmat(zahhak_pos, n_active, 1) - old_pos);
        
        % Phase 3: Social Learning (SL) - Peer Interaction
        self_idx = active_indices(:);
        same = (partner_idx == self_idx);
        while any(same)
            partner_idx(same) = randi(popSize, sum(same), 1);
            same = (partner_idx == self_idx);
        end
        r_vec2 = rand(n_active, dim);
        SL = r_vec2 .* (solutions(partner_idx, :) - old_pos);
        
        % Stochastically assign DM or SL updates
        A = zeros(n_active, dim);
        use_DM = (p > 0.5);
        A(use_DM, :)  = DM(use_DM, :);
        A(~use_DM, :) = SL(~use_DM, :);
        
        % Generate and Repair Candidates
        new_pos = old_pos + guided + A;
        new_pos = BoundReflect(new_pos, lb_vec, ub_vec);
        
        % Evaluate Candidates and Apply Greedy Selection
        for k = 1:n_active
            if FE >= fe_limit, break; end
            cand = new_pos(k, :);
            f_cand = func(cand);
            FE = FE + 1;
            
            if f_cand < old_fit(k)
                idxk = active_indices(k);
                solutions(idxk, :) = cand;
                fitnesses(idxk)   = f_cand;
                
                % Update Global Best
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

        %% ================= BUZJANI TILING STRATEGY (BTS) =================
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
        
        % Subspace Selection
        if dim == 1
            dims_selected = 1;
            k_dims = 1;
        else
            dims_perm = randperm(dim);
            dims_selected = dims_perm(1:2);
            k_dims = 2;
        end
        
        if k_dims == 2
            % Construct 2D Orthonormal Basis via Gram-Schmidt
            r1 = randn(2,1);
            r2 = randn(2,1);
            u = r1 / (norm(r1) + 1e-12);
            v = r2 - (dot(r2,u) * u);
            v = v / (norm(v) + 1e-12);
            
            phi = 2*pi*rand();
            hex_angles = (0:5) * (pi/3);
            
            best_hex_fit = base_fit;
            best_hex_pos = base_pos;
            
            % Hexagonal Probing
            for a = hex_angles
                if FE >= fe_limit, break; end
                ang = a + phi;
                offset2 = radius * (cos(ang) * u + sin(ang) * v); 
                
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
                solutions(min_idx, :) = zahhak_pos;
                fitnesses(min_idx)   = zahhak_fit;
                
                if FE <= fe_limit
                    convergence_curve(FE) = zahhak_fit;
                end
            else
                % Fallback: Stochastic Axis-Aligned Step
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
                    solutions(min_idx, :) = zahhak_pos;
                    fitnesses(min_idx)   = zahhak_fit;
                    convergence_curve(FE) = zahhak_fit;
                end
            end
        else
            % 1-Dimensional Edge Case Handling
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
                solutions(min_idx, :) = zahhak_pos;
                fitnesses(min_idx)   = zahhak_fit;
                convergence_curve(FE) = zahhak_fit;
            end
        end
    end
    
    % Trim preallocated indices
    convergence_curve = convergence_curve(1:FE);
end

% =========================================================================
% Constraint Handling: Reflection Boundary Repair
% =========================================================================
function X = BoundReflect(X, lb_vec, ub_vec)
    lb_vec = lb_vec(:)';
    ub_vec = ub_vec(:)';
    n = size(X, 1);
    LB = repmat(lb_vec, n, 1);
    UB = repmat(ub_vec, n, 1);
    
    % Primary Reflection
    idxL = X < LB;
    X(idxL) = 2 * LB(idxL) - X(idxL);
    idxU = X > UB;
    X(idxU) = 2 * UB(idxU) - X(idxU);
    
    % Fold extreme overshoots to guarantee boundaries
    range = (UB - LB) + eps;
    X = LB + mod(X - LB, 2 * range);
    
    idxU2 = X > UB;
    X(idxU2) = UB(idxU2) - (X(idxU2) - UB(idxU2));
    
    % Absolute safety clamp (catches numerical precision edge-cases)
    X = max(min(X, UB), LB);
end