% =========================================================================
% Wireless Sensor Network (WSN) Coverage Optimization using Metaheuristics
% Enhanced with the Buzjani Tiling Strategy (BTS)
% =========================================================================
% This script optimizes the 2D deployment of homogeneous sensors to maximize
% the Area of Interest (RoI) coverage. It compares base metaheuristic 
% algorithms against their BTS-enhanced variants under a strict 
% Function Evaluation (FE) budget.
% =========================================================================

function WSN_Coverage_Integration_Buzjani()
    clc; clear; close all;
    
    % ---------------- EXPERIMENT SETTINGS ----------------
    N_RUNS     = 30;           % Independent runs for statistical significance
    POP_SIZE   = 40;           % Population size (Swarm size)
    FE_LIMIT   = 5000;         % Max Function Evaluations (lower for real-world problems)
    LS_RATIO   = 0.10;         % 10% FE budget reserved strictly for BTS local search
    baseSeed   = 2026;         % Base random seed for exact reproducibility
    
    % ---------------- WSN PROBLEM PARAMETERS ----------------
    SpaceWidth  = 50;          % RoI Width (meters)
    SpaceHeight = 50;          % RoI Height (meters)
    NumSensors  = 30;          % Total number of deployable sensors
    SensingRadius = 5;         % Detection radius of each sensor (meters)
    GridResolution = 1.0;      % Discretization grid step for coverage calculation
    
    % The optimization space dimension is 2D coordinates (x, y) per sensor
    TARGET_DIM = NumSensors * 2; 
    LB_VAL = 0;                % Lower bound of the deployment area
    UB_VAL = SpaceWidth;       % Upper bound (Assuming square RoI)
    SAVE_FILE  = 'WSN_Integration_Buzjani.txt';
    
    % ---------------- INITIALIZE OUTPUT LOGGING ----------------
    fid = fopen(SAVE_FILE, 'w');
    fprintf(fid, 'WSN Coverage Optimization | Sensors=%d | R=%d | Area=%dx%d | FE=%d\n\n', ...
        NumSensors, SensingRadius, SpaceWidth, SpaceHeight, FE_LIMIT);
    fprintf(fid, '| Alg | Mode | Best Coverage(%%) | Mean Cov(%%) | Std | Time(s) |\n');
    fprintf(fid, '|-----|------|------------------|--------------|-----|---------|\n');
    
    % ---------------- CONSTRUCT COST FUNCTION ----------------
    % Metaheuristics minimize cost. Therefore, Cost = -(Coverage Rate)
    wsn_func = @(SensorCoordinates) WSN_Coverage_Cost(SensorCoordinates, ...
        NumSensors, SpaceWidth, SpaceHeight, SensingRadius, GridResolution);
    
    % Wrapper to strictly enforce boundaries during evaluations
    safe_func = @(SensorCoordinates) SafeEval(wsn_func, SensorCoordinates, ...
        LB_VAL * ones(1, TARGET_DIM), UB_VAL * ones(1, TARGET_DIM));
        
    % ---------------- ALGORITHM CONFIGURATION ----------------
    % Uncomment baselines as needed. Format: {'Name', @function_handle}
    algs = { ...
        'SO',     @(d,lb,ub,fe,N,f) SO_Budgeted(d,lb,ub,fe,N,f);
        'GWO',    @(d,lb,ub,fe,N,f) GWO_Budgeted(d,lb,ub,fe,N,f);
        'TLBO',   @(d,lb,ub,fe,N,f) TLBO_Budgeted(d,lb,ub,fe,N,f);
        'SCA',    @(d,lb,ub,fe,N,f) SCA_Budgeted(d,lb,ub,fe,N,f);
        'MVO',    @(d,lb,ub,fe,N,f) MVO_Budgeted(d,lb,ub,fe,N,f);
        'TJO',    @(d,lb,ub,fe,N,f) TJO_Budgeted(d,lb,ub,fe,N,f);
        'GPC',    @(d,lb,ub,fe,N,f) GPC_Budgeted(d,lb,ub,fe,N,f);
        'PPO',    @(d,lb,ub,fe,N,f) PPO_Budgeted(d,lb,ub,fe,N,f);
        'CAO',    @(d,lb,ub,fe,N,f) CAO_Budgeted(d,lb,ub,fe,N,f);
        'HIO',    @(d,lb,ub,fe,N,f) HIO_Budgeted(d,lb,ub,fe,N,f);
        'ACSBOA', @(d,lb,ub,fe,N,f) ACSBOA_Budgeted(d,lb,ub,fe,N,f);
    };

    GlobalBestPos = [];
    GlobalBestCov = -inf;
    BestAlgName = '';
    
    fprintf('Initiating WSN Coverage Optimization Framework...\n');
    
    for a = 1:size(algs, 1)
        algName = algs{a, 1};
        algFun  = algs{a, 2};
        
        scores_base = zeros(N_RUNS, 1);
        scores_buz  = zeros(N_RUNS, 1);
        times_base  = zeros(N_RUNS, 1);
        times_buz   = zeros(N_RUNS, 1);
        
        for run = 1:N_RUNS
            % Dynamic seed generation for isolated, reproducible runs
            seed_run = baseSeed + 100*a + run;
            
            % ---------------- BASE ALGORITHM (100% FE_LIMIT) ----------------
            rng(seed_run, 'twister');
            t0 = tic;
            [bestx_base, bestf_base, ~, ~] = algFun(TARGET_DIM, LB_VAL, UB_VAL, FE_LIMIT, POP_SIZE, safe_func);
            times_base(run)  = toc(t0);
            scores_base(run) = -bestf_base * 100; % Revert negative cost to positive Percentage
            
            % ---------------- BTS ENHANCED (90% Base + 10% BTS) ----------------
            rng(seed_run, 'twister');
            t1 = tic;
            [bestx_buz, bestf_buz, ~, ~] = RunWithBuzjaniEndLS( ...
                algFun, TARGET_DIM, LB_VAL, UB_VAL, FE_LIMIT, POP_SIZE, safe_func, LS_RATIO);
            times_buz(run)  = toc(t1);
            scores_buz(run) = -bestf_buz * 100;   % Revert negative cost to positive Percentage
            
            % Track the absolute best configuration for the final topology plot
            if scores_buz(run) > GlobalBestCov
                GlobalBestCov = scores_buz(run);
                GlobalBestPos = bestx_buz;
                BestAlgName = [algName ' +BTS'];
            end
            if scores_base(run) > GlobalBestCov
                GlobalBestCov = scores_base(run);
                GlobalBestPos = bestx_base;
                BestAlgName = [algName ' (Base)'];
            end
        end
        
        % Print statistical summaries to log
        PrintStatsWSN(fid, algName, 'Base', scores_base, times_base);
        PrintStatsWSN(fid, algName, '+BTS', scores_buz,  times_buz);
        fprintf('Successfully evaluated framework: %s\n', algName);
    end
    
    fclose(fid);
    disp(['Experiment complete. Results saved to: ' SAVE_FILE]);
    
    % ---------------- PLOT OPTIMAL TOPOLOGY ----------------
    if ~isempty(GlobalBestPos)
        PlotWSN(GlobalBestPos, NumSensors, SpaceWidth, SpaceHeight, SensingRadius, BestAlgName, GlobalBestCov);
    end
end

% =========================================================================
% WSN Cost Function: Computes Negative Coverage Rate
% =========================================================================
function cost = WSN_Coverage_Cost(SensorCoordinates, NumSensors, W, H, SensingRadius, GridRes)
    % Reshape 1D coordinate array to (N x 2) matrix [x1, y1; x2, y2; ...]
    sensors = reshape(SensorCoordinates, [2, NumSensors])';  
    
    % Construct the discretized observation grid
    gx = 0:GridRes:W;
    gy = 0:GridRes:H;
    [GX, GY] = meshgrid(gx, gy);
    GridPoints = [GX(:), GY(:)];  
    
    R2 = SensingRadius^2;
    
    % High-speed Vectorized Distance Calculation (Implicit Expansion)
    % dx and dy result in (nPoints x NumSensors) matrices instantly
    dx = GridPoints(:,1) - sensors(:,1)'; 
    dy = GridPoints(:,2) - sensors(:,2)';
    dist2 = dx.^2 + dy.^2;
    
    % A discrete grid point is considered 'covered' if ANY sensor is within range
    covered = any(dist2 <= R2, 2);
    
    % Calculate Total Coverage Percentage
    coverageRate = mean(covered);   
    
    % Optimization seeks to minimize, so we return the negative rate
    cost = -coverageRate;           
end

% =========================================================================
% Helper: Print Statistical Formatted Output
% =========================================================================
function PrintStatsWSN(fid, algName, modeName, scores, times)
    best_val = max(scores); 
    mean_val = mean(scores);
    std_val  = std(scores);
    t_mean = mean(times);
    fprintf(fid, '| %-6s | %-6s | %06.2f%%           | %06.2f%%       | %05.2f | %.4f |\n', ...
        algName, modeName, best_val, mean_val, std_val, t_mean);
end

% =========================================================================
% Helper: Visualize the Final WSN Topology
% =========================================================================
function PlotWSN(BestSensorCoordinates, NumSensors, W, H, SensingRadius, TitleStr, CovVal)
    sensors = reshape(BestSensorCoordinates, [2, NumSensors])';
    figure('Color','w', 'Name', 'Optimal WSN Deployment Topology');
    hold on; box on; axis equal;
    xlim([0 W]); ylim([0 H]);
    
    % Plot sensing radii (coverage discs)
    theta = linspace(0, 2*pi, 50);
    for i = 1:NumSensors
        cx = sensors(i,1) + SensingRadius*cos(theta);
        cy = sensors(i,2) + SensingRadius*sin(theta);
        fill(cx, cy, [0.8 0.8 1], 'EdgeColor', 'b', 'FaceAlpha', 0.3);
        plot(sensors(i,1), sensors(i,2), 'r.', 'MarkerSize', 12); % Sensor nodes
    end
    
    title(sprintf('%s Topology - Max Coverage Achieved: %.2f%%', TitleStr, CovVal), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Environment Width (m)'); 
    ylabel('Environment Height (m)');
    grid on;
    hold off;
end

% =========================================================================
% Helper: Safe Evaluation (Boundary Clamp)
% =========================================================================
function y = SafeEval(fun, SensorCoordinates, lb_vec, ub_vec)
    SensorCoordinates = SensorCoordinates(:)';                              
    SensorCoordinates = max(min(SensorCoordinates, ub_vec), lb_vec);        
    y = fun(SensorCoordinates);
end

% =========================================================================
% Core Execution: Run Base Optimizer Followed by BTS Refinement
% =========================================================================
function [bestx, bestf, curve_full, FE_full] = RunWithBuzjaniEndLS( ...
    alg_fun, dim, lb, ub, fe_limit, popSize, fobj, ls_ratio)
    
    if nargin < 8, ls_ratio = 0.10; end
    
    % 1. Execute Base Algorithm (e.g., 90% FE Budget)
    fe_base = max(1, floor(fe_limit * (1 - ls_ratio)));
    [bestx0, bestf0, curve0, FE0] = alg_fun(dim, lb, ub, fe_base, popSize, fobj);
    
    % 2. Verify Remaining Computational Budget
    fe_rem = fe_limit - FE0;
    if fe_rem <= 0
        bestx = bestx0; bestf = bestf0;
        curve_full = curve0; FE_full = FE0;
        return;
    end
    
    % 3. Execute Buzjani Tiling Strategy (BTS) (e.g., 10% FE Budget)
    if isscalar(lb), lb_vec = lb*ones(1,dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub*ones(1,dim); else, ub_vec = ub(:)'; end
    
    [bestx, bestf, curveLS, FE_LS] = BuzjaniLocalSearch( ...
        bestx0, bestf0, fe_rem, fobj, lb_vec, ub_vec);
        
    % Concatenate Convergence Histories
    curve_full = [curve0(:)' curveLS(:)'];
    FE_full    = FE0 + FE_LS;
end

% =========================================================================
% Buzjani Tiling Strategy (BTS) - Local Search Operator
% =========================================================================
function [bestx, bestf, curve, FE] = BuzjaniLocalSearch(bestx, bestf, fe_budget, fobj, lb_vec, ub_vec)
    FE = 0;
    curve = inf(1, fe_budget); 
    dim = numel(bestx);
    Delta = mean(ub_vec - lb_vec); 
    
    % Control Parameters 
    omega = 0.10;   
    
    % Main Exploitation Loop
    while FE < fe_budget
        
        % 1. Dynamic Radius Computation (Shrinking Trust Region)
        progress = FE / fe_budget; 
        radius = (Delta * omega * (1 - progress)^3) + 1e-12; 
        
        base_pos = bestx;
        base_fit = bestf;
        
        % 2. Subspace Selection (k = 2)
        if dim > 1
            dims_selected = randperm(dim, 2);
            k_dims = 2;
        else
            dims_selected = 1;
            k_dims = 1;
        end
        
        improved = false;
        
        % Phase A: Hexagonal Tiling (2D Subspace)
        if k_dims == 2
            
            % Construct Orthonormal Basis (Gram-Schmidt process)
            r1 = randn(2,1); 
            r2 = randn(2,1);
            
            u = r1 / (norm(r1) + eps);
            v_raw = r2 - (dot(r2, u) * u);
            v = v_raw / (norm(v_raw) + eps);
            
            phi = 2 * pi * rand();
            hex_angles = (0:5) * (pi/3);
            
            best_hex_fit = base_fit;
            best_hex_pos = base_pos;
            
            for a = hex_angles
                if FE >= fe_budget, break; end
                
                ang = a + phi;
                offset_2d = radius * (cos(ang)*u + sin(ang)*v);
                
                cand = base_pos;
                cand(dims_selected(1)) = cand(dims_selected(1)) + offset_2d(1);
                cand(dims_selected(2)) = cand(dims_selected(2)) + offset_2d(2);
                
                cand = max(min(cand, ub_vec), lb_vec);
                f_cand = fobj(cand);
                FE = FE + 1;
                
                if f_cand < best_hex_fit
                    best_hex_fit = f_cand;
                    best_hex_pos = cand;
                end
                curve(FE) = min(bestf, best_hex_fit);
            end
            
            if best_hex_fit < bestf
                bestf = best_hex_fit;
                bestx = best_hex_pos;
                improved = true;
            end
            
            % Phase B: Fallback Stochastic Axis-Aligned Step
            if ~improved
                best_ax_fit = bestf;
                best_ax_pos = bestx;
                
                for d = dims_selected
                    for s = [-1, 1] 
                        if FE >= fe_budget, break; end
                        
                        rho = rand(); 
                        cand = bestx;
                        cand(d) = cand(d) + (s * rho * radius);
                        
                        cand = max(min(cand, ub_vec), lb_vec);
                        f_cand = fobj(cand);
                        FE = FE + 1;
                        
                        if f_cand < best_ax_fit
                            best_ax_fit = f_cand;
                            best_ax_pos = cand;
                        end
                        curve(FE) = min(bestf, best_ax_fit);
                    end
                    if FE >= fe_budget, break; end
                end
                
                if best_ax_fit < bestf
                    bestf = best_ax_fit;
                    bestx = best_ax_pos;
                end
            end
            
        % Phase C: Edge Case (1-Dimensional Problem)
        else 
            best_1d_fit = bestf;
            best_1d_pos = bestx;
            
            for s = [-1, 1]
                if FE >= fe_budget, break; end
                rho = rand();
                cand = bestx;
                cand(1) = cand(1) + (s * rho * radius);
                
                cand = max(min(cand, ub_vec), lb_vec);
                f_cand = fobj(cand);
                FE = FE + 1;
                
                if f_cand < best_1d_fit
                    best_1d_fit = f_cand;
                    best_1d_pos = cand;
                end
                curve(FE) = min(bestf, best_1d_fit);
            end
            
            if best_1d_fit < bestf
                bestf = best_1d_fit;
                bestx = best_1d_pos;
            end
        end
    end 
    curve = curve(1:FE);
end

% =========================================================================
% ALGORITHM REPOSITORIES (Base Optimizers)
% Note: Core logic maintained strictly matching benchmark specifications.
% =========================================================================

% --- SO Budgeted ---
function [Xfood, fval, convergence_curve, FE] = SO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    maxT = floor((fe_limit - N) / N); if maxT < 0, maxT = 0; end
    convergence_curve = inf(1, fe_limit);
    best_so_far = inf;
    X = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    fitness = inf(N,1);
    for i = 1:N
        if FE >= fe_limit, break; end
        X(i,:) = max(min(X(i,:), ub_vec), lb_vec);
        FE = FE + 1;
        fitness(i) = fobj(X(i,:));
        best_so_far = min(best_so_far, fitness(i));
        convergence_curve(FE) = best_so_far;
    end
    if FE >= fe_limit, [fval, gbest] = min(fitness); Xfood = X(gbest,:); return; end
    [fval, gbest] = min(fitness); Xfood = X(gbest,:);
    Nm = round(N/2); Nf = N - Nm;
    Xm = X(1:Nm,:); Xf = X(Nm+1:N,:);
    fitness_m = fitness(1:Nm); fitness_f = fitness(Nm+1:N);
    [fitnessBest_m, g1] = min(fitness_m); Xbest_m = Xm(g1,:);
    [fitnessBest_f, g2] = min(fitness_f); Xbest_f = Xf(g2,:);
    vec_flag=[1,-1]; C1=0.5; C2=0.05; C3=2; Threshold=0.25; Thresold2=0.6;
    
    for t = 1:maxT
        if FE >= fe_limit, break; end
        Temp = exp(-(t/maxT)); Q = C1*exp((t-maxT)/maxT); if Q>1, Q=1; end
        Xnewm = Xm; Xnewf = Xf;
        if Q < Threshold
            for i=1:Nm, Xnewm(i,:) = Xm(randi(Nm),:) + vec_flag(randi(2))*C2*exp(-fitness_m(randi(Nm))/(fitness_m(i)+eps))*(lb_vec+rand(1,dim).*(ub_vec-lb_vec)); end
            for i=1:Nf, Xnewf(i,:) = Xf(randi(Nf),:) + vec_flag(randi(2))*C2*exp(-fitness_f(randi(Nf))/(fitness_f(i)+eps))*(lb_vec+rand(1,dim).*(ub_vec-lb_vec)); end
        else
            if Temp > Thresold2
                 for i=1:Nm, Xnewm(i,:) = Xfood + C3*vec_flag(randi(2))*Temp*rand*(Xfood-Xm(i,:)); end
                 for i=1:Nf, Xnewf(i,:) = Xfood + C3*vec_flag(randi(2))*Temp*rand*(Xfood-Xf(i,:)); end
            else
                if rand>0.6
                    for i=1:Nm, Xnewm(i,:) = Xm(i,:) + C3*exp(-fitnessBest_f/(fitness_m(i)+eps))*rand*(Q*Xbest_f-Xm(i,:)); end
                    for i=1:Nf, Xnewf(i,:) = Xf(i,:) + C3*exp(-fitnessBest_m/(fitness_f(i)+eps))*rand*(Q*Xbest_m-Xf(i,:)); end
                else
                    for i=1:Nm, Xnewm(i,:) = Xm(i,:) + C3*rand*exp(-fitness_f(min(i,Nf))/(fitness_m(i)+eps))*(Q*Xf(min(i,Nf),:)-Xm(i,:)); end
                    for i=1:Nf, Xnewf(i,:) = Xf(i,:) + C3*rand*exp(-fitness_m(min(i,Nm))/(fitness_f(i)+eps))*(Q*Xm(min(i,Nm),:)-Xf(i,:)); end
                end
            end
        end
        % Update
        for j=1:Nm
            if FE>=fe_limit, break; end
            Xnewm(j,:) = max(min(Xnewm(j,:), ub_vec), lb_vec);
            FE=FE+1; fit=fobj(Xnewm(j,:));
            if fit<fitness_m(j), fitness_m(j)=fit; Xm(j,:)=Xnewm(j,:); end
            best_so_far = min(best_so_far, min([fitness_m; fitness_f])); convergence_curve(FE)=best_so_far;
        end
        for j=1:Nf
             if FE>=fe_limit, break; end
             Xnewf(j,:) = max(min(Xnewf(j,:), ub_vec), lb_vec);
             FE=FE+1; fit=fobj(Xnewf(j,:));
             if fit<fitness_f(j), fitness_f(j)=fit; Xf(j,:)=Xnewf(j,:); end
             best_so_far = min(best_so_far, min([fitness_m; fitness_f])); convergence_curve(FE)=best_so_far;
        end
        [fbm, gm] = min(fitness_m); if fbm<fitnessBest_m, fitnessBest_m=fbm; Xbest_m=Xm(gm,:); end
        [fbf, gf] = min(fitness_f); if fbf<fitnessBest_f, fitnessBest_f=fbf; Xbest_f=Xf(gf,:); end
        if fitnessBest_m < fitnessBest_f, fval=fitnessBest_m; Xfood=Xbest_m; else, fval=fitnessBest_f; Xfood=Xbest_f; end
    end
    convergence_curve=convergence_curve(1:FE);
end

% --- GWO Budgeted ---
function [Alpha_pos, Alpha_score, ConvergenceCurve, FE] = GWO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    Max_iter = floor(fe_limit / N); if Max_iter < 1, Max_iter = 1; end
    Alpha_pos=zeros(1,dim); Alpha_score=inf; Beta_pos=zeros(1,dim); Beta_score=inf; Delta_pos=zeros(1,dim); Delta_score=inf;
    Positions = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    ConvergenceCurve = zeros(1, fe_limit);
    for l=1:Max_iter
        for i=1:N
            if FE>=fe_limit, break; end
            Positions(i,:) = max(min(Positions(i,:), ub_vec), lb_vec);
            FE=FE+1; fitness=fobj(Positions(i,:));
            if fitness<Alpha_score, Alpha_score=fitness; Alpha_pos=Positions(i,:);
            elseif fitness<Beta_score, Beta_score=fitness; Beta_pos=Positions(i,:);
            elseif fitness<Delta_score, Delta_score=fitness; Delta_pos=Positions(i,:); end
            ConvergenceCurve(FE) = Alpha_score;
        end
        if FE>=fe_limit, break; end
        a=2-l*((2)/Max_iter);
        for i=1:N
            for j=1:dim
                X1=Alpha_pos(j)- (2*a*rand-a)*abs(2*rand*Alpha_pos(j)-Positions(i,j));
                X2=Beta_pos(j) - (2*a*rand-a)*abs(2*rand*Beta_pos(j)-Positions(i,j));
                X3=Delta_pos(j)- (2*a*rand-a)*abs(2*rand*Delta_pos(j)-Positions(i,j));
                Positions(i,j)=(X1+X2+X3)/3;
            end
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- SCA Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = SCA_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb*ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub*ones(1, dim); else, ub_vec = ub(:)'; end
    MaxIter = floor((fe_limit - N) / N); if MaxIter<0, MaxIter=0; end
    ConvergenceCurve = zeros(1, fe_limit);
    X = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    bestf = inf; bestx = X(1,:);
    for i=1:N
        if FE>=fe_limit, break; end
        X(i,:) = max(min(X(i,:), ub_vec), lb_vec);
        FE=FE+1; fit=fobj(X(i,:));
        if fit<bestf, bestf=fit; bestx=X(i,:); end
        ConvergenceCurve(FE)=bestf;
    end
    for t=1:MaxIter
        if FE>=fe_limit, break; end
        r1 = 2 - t*(2/MaxIter);
        for i=1:N
            for j=1:dim
                if rand<0.5, X(i,j)=X(i,j)+(r1*sin(rand*2*pi)*abs(2*rand*bestx(j)-X(i,j)));
                else, X(i,j)=X(i,j)+(r1*cos(rand*2*pi)*abs(2*rand*bestx(j)-X(i,j))); end
            end
        end
        for i=1:N
            if FE>=fe_limit, break; end
            X(i,:) = max(min(X(i,:), ub_vec), lb_vec);
            FE=FE+1; fit=fobj(X(i,:));
            if fit<bestf, bestf=fit; bestx=X(i,:); end
            ConvergenceCurve(FE)=bestf;
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- MVO Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = MVO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb*ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub*ones(1, dim); else, ub_vec = ub(:)'; end
    MaxIter = max(1, floor((fe_limit - N) / N));
    ConvergenceCurve = inf(1, fe_limit);
    Universes = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    Inflation_rates = inf(N,1);
    bestf = inf;
    bestx = Universes(1,:);
    for i = 1:N
        if FE >= fe_limit, break; end
        Universes(i,:) = max(min(Universes(i,:), ub_vec), lb_vec);
        FE = FE + 1;
        Inflation_rates(i) = fobj(Universes(i,:));
        if Inflation_rates(i) < bestf
            bestf = Inflation_rates(i);
            bestx = Universes(i,:);
        end
        ConvergenceCurve(FE) = bestf;
    end
    if FE >= fe_limit
        ConvergenceCurve = ConvergenceCurve(1:FE);
        return;
    end
    WEP_Min = 0.2; WEP_Max = 1.0;
    for Time = 1:MaxIter
        if FE >= fe_limit, break; end
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / MaxIter);
        TDR = 1 - (Time^(1/6) / MaxIter^(1/6));  
        [sorted_rates, idx] = sort(Inflation_rates, 'ascend');
        Sorted_universes = Universes(idx,:);
        Universes(1,:) = Sorted_universes(1,:);
        bestF = sorted_rates(1);
        invFit = 1 ./ (sorted_rates - bestF + eps); 
        prob = invFit ./ sum(invFit);
        cdf = cumsum(prob);                            
        for i = 2:N
            for j = 1:dim
                r = rand;
                white = find(cdf >= r, 1, 'first');
                Universes(i,j) = Sorted_universes(white, j);
                if rand < WEP
                    step = (ub_vec(j) - lb_vec(j)) * rand;
                    if rand < 0.5
                        Universes(i,j) = bestx(j) + TDR * step;
                    else
                        Universes(i,j) = bestx(j) - TDR * step;
                    end
                end
            end
        end
        for i = 1:N
            if FE >= fe_limit, break; end
            Universes(i,:) = max(min(Universes(i,:), ub_vec), lb_vec);
            FE = FE + 1;
            Inflation_rates(i) = fobj(Universes(i,:));
            if Inflation_rates(i) < bestf
                bestf = Inflation_rates(i);
                bestx = Universes(i,:);
            end
            ConvergenceCurve(FE) = bestf;
        end
    end
    ConvergenceCurve = ConvergenceCurve(1:FE);
end

% --- TLBO Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = TLBO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb*ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub*ones(1, dim); else, ub_vec = ub(:)'; end
    ConvergenceCurve = zeros(1, fe_limit);
    pop = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    costs = zeros(N,1);
    for i=1:N
        if FE>=fe_limit, break; end
        FE=FE+1; costs(i)=fobj(pop(i,:));
        ConvergenceCurve(FE)=min(costs(1:i));
    end
    [bestf, idx] = min(costs); bestx = pop(idx,:);
    while FE < fe_limit
        Mean = mean(pop);
        [~, t_idx] = min(costs); Teacher = pop(t_idx,:);
        % Teaching Phase
        for i=1:N
            if FE>=fe_limit, break; end
            TF = randi([1 2]);
            newsol = pop(i,:) + rand(1,dim).*(Teacher - TF*Mean);
            newsol = max(min(newsol, ub_vec), lb_vec);
            FE=FE+1; newcost = fobj(newsol);
            if newcost < costs(i), pop(i,:)=newsol; costs(i)=newcost; end
            if newcost < bestf, bestf=newcost; bestx=newsol; end
            ConvergenceCurve(FE)=bestf;
        end
        % Learning Phase
        for i=1:N
             if FE>=fe_limit, break; end
             peer = randi(N); while peer==i, peer=randi(N); end
             if costs(i) < costs(peer), step = pop(i,:) - pop(peer,:); else, step = pop(peer,:) - pop(i,:); end
             newsol = pop(i,:) + rand(1,dim).*step;
             newsol = max(min(newsol, ub_vec), lb_vec);
             FE=FE+1; newcost = fobj(newsol);
             if newcost < costs(i), pop(i,:)=newsol; costs(i)=newcost; end
             if newcost < bestf, bestf=newcost; bestx=newsol; end
             ConvergenceCurve(FE)=bestf;
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- TJO Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = TJO_Budgeted(nvars, lb, ub, fe_limit, N, fun)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, nvars); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, nvars); else, ub_vec = ub(:)'; end
    ConvergenceCurve = zeros(1, fe_limit);
    x = repmat(lb_vec, N, 1) + rand(N, nvars) .* repmat(ub_vec - lb_vec, N, 1);
    f = zeros(N,1);
    for i=1:N
        if FE >= fe_limit, break; end
        f(i) = fun(x(i,:)); FE = FE + 1; ConvergenceCurve(FE) = min(f(1:i));
    end
    [bestf, idx] = min(f); bestx = x(idx,:);
    FlockX = x; FlockF = f;
    while FE < fe_limit
        r = FE / fe_limit; c_t = 0.1 + 0.4*r;
        BestX = (1-r).*FlockX + r.*bestx;
        x_new = BestX + c_t*rand(N, nvars).*(repmat(lb_vec,N,1) + rand(N,nvars).*(repmat(ub_vec,N,1)-repmat(lb_vec,N,1)));
        for i=1:N
            if FE>=fe_limit, break; end
            x_new(i,:) = max(min(x_new(i,:), ub_vec), lb_vec);
            fi = fun(x_new(i,:)); FE=FE+1;
            if fi<FlockF(i), FlockF(i)=fi; FlockX(i,:)=x_new(i,:); end
            if fi<bestf, bestf=fi; bestx=x_new(i,:); end
            ConvergenceCurve(FE)=bestf;
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- GPC Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = GPC_Budgeted(dim, lb, ub, fe_limit, nPop, CostFunction)
    FE=0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    pos = repmat(lb_vec, nPop, 1) + rand(nPop, dim).*repmat(ub_vec-lb_vec, nPop, 1);
    costs = zeros(nPop, 1);
    ConvergenceCurve = zeros(1, fe_limit);
    for i=1:nPop
        if FE>=fe_limit, break; end
        costs(i)=CostFunction(pos(i,:)); FE=FE+1; ConvergenceCurve(FE)=min(costs(1:i));
    end
    [bestf, idx] = min(costs); bestx = pos(idx,:);
    G=9.8; Tetha=14;
    while FE < fe_limit
        for i=1:nPop
            if FE>=fe_limit, break; end
            V0 = rand; Mu = 1+9*rand;
            d = (V0^2)/((2*G)*(sind(Tetha)+(Mu*cosd(Tetha))));
            cand = pos(i,:) + d*rand(1,dim); 
            cand = max(min(cand, ub_vec), lb_vec);
            newcost = CostFunction(cand); FE=FE+1;
            if newcost < costs(i), pos(i,:)=cand; costs(i)=newcost; end
            if newcost < bestf, bestf=newcost; bestx=cand; end
            ConvergenceCurve(FE)=bestf;
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- PPO Budgeted ---
function [xi, bestf, ConvergenceCurve, FE] = PPO_Budgeted(d, lb, ub, fe_limit, N, fun)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, d); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, d); else, ub_vec = ub(:)'; end
    ConvergenceCurve = zeros(1,fe_limit);
    X = repmat(lb_vec, N, 1) + rand(N, d) .* repmat(ub_vec - lb_vec, N, 1);
    f = inf(N,1); bestf = inf; xi = X(1,:);
    for i=1:N
        if FE>=fe_limit, break; end
        f(i)=fun(X(i,:)); FE=FE+1;
        if f(i)<bestf, bestf=f(i); xi=X(i,:); end
        ConvergenceCurve(FE)=bestf;
    end
    while FE < fe_limit
        index = randperm(N); Y = X(index,:);
        F = max(f) + min(f) - f; E = F/(max(F)+eps);
        for i=1:N
           if FE>=fe_limit, break; end
           v = E(i).*sqrt((X(i,:)-Y(i,:)).^2);
           X(i,:) = Y(i,:) + cos(rand(1,d)*pi).*v;
           X(i,:) = max(min(X(i,:), ub_vec), lb_vec);
           f(i)=fun(X(i,:)); FE=FE+1;
           if f(i)<bestf, bestf=f(i); xi=X(i,:); end
           ConvergenceCurve(FE)=bestf;
        end
    end
    ConvergenceCurve=ConvergenceCurve(1:FE);
end

% --- CAO Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = CAO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    
    ConvergenceCurve = zeros(1, fe_limit);
    X = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    fit = inf(1, N);
    
    bestf = inf;
    bestx = X(1, :);
    
    for i = 1:N
        if FE >= fe_limit, break; end
        X(i, :) = max(min(X(i, :), ub_vec), lb_vec);
        fit(i) = fobj(X(i, :));
        FE = FE + 1;
        
        if fit(i) < bestf
            bestf = fit(i);
            bestx = X(i, :);
        end
        ConvergenceCurve(FE) = bestf;
    end
    
    while FE < fe_limit
        for j = 1:N
            if FE >= fe_limit, break; end
            
            % 1. Competition Phase
            k = randi(N);
            while k == j && N > 1
                k = randi(N);
            end
            
            MI = (X(j, :) + X(k, :)) / 2; 
            I1 = randi([1, 2]); 
            I2 = randi([1, 2]); 
            
            X_j_new = X(j, :) - rand(1, dim) .* ((MI * I1) - bestx); 
            X_k_new = X(k, :) - rand(1, dim) .* ((MI * I2) - bestx); 
            
            % Evaluate new j
            X_j_new = max(min(X_j_new, ub_vec), lb_vec);
            f_j_new = fobj(X_j_new);
            FE = FE + 1;
            if f_j_new < fit(j)
                X(j, :) = X_j_new;
                fit(j) = f_j_new;
                if f_j_new < bestf
                    bestf = f_j_new;
                    bestx = X_j_new;
                end
            end
            ConvergenceCurve(FE) = bestf;
            if FE >= fe_limit, break; end
            
            % Evaluate new k
            X_k_new = max(min(X_k_new, ub_vec), lb_vec);
            f_k_new = fobj(X_k_new);
            FE = FE + 1;
            if f_k_new < fit(k)
                X(k, :) = X_k_new;
                fit(k) = f_k_new;
                if f_k_new < bestf
                    bestf = f_k_new;
                    bestx = X_k_new;
                end
            end
            ConvergenceCurve(FE) = bestf;
            if FE >= fe_limit, break; end
            
            % 2. Amensalism Phase
            k_amen = randi(N);
            while k_amen == j && N > 1
                k_amen = randi(N);
            end
            
            rand_amen = 0.5 + 0.5 * rand(1, dim); 
            X_amen_new = X(j, :) - rand_amen .* (X(k_amen, :) - bestx); 
            
            X_amen_new = max(min(X_amen_new, ub_vec), lb_vec);
            f_amen_new = fobj(X_amen_new);
            FE = FE + 1;
            
            if f_amen_new < fit(j)
                X(j, :) = X_amen_new;
                fit(j) = f_amen_new;
                if f_amen_new < bestf
                    bestf = f_amen_new;
                    bestx = X_amen_new;
                end
            end
            ConvergenceCurve(FE) = bestf;
        end
    end
    ConvergenceCurve = ConvergenceCurve(1:FE);
end

% --- HIO Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = HIO_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    
    ConvergenceCurve = zeros(1, fe_limit);
    X = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    V = zeros(N, dim);
    fit = inf(1, N);
    
    bestf = inf;
    bestx = X(1, :);
    
    for i = 1:N
        if FE >= fe_limit, break; end
        X(i, :) = max(min(X(i, :), ub_vec), lb_vec);
        fit(i) = fobj(X(i, :));
        FE = FE + 1;
        if fit(i) < bestf
            bestf = fit(i);
            bestx = X(i, :);
        end
        ConvergenceCurve(FE) = bestf;
    end
    
    % HIO Parameters
    BU = 10; 
    num_clusters = min(10, N); 
    w = 0.5; 
    beta_levy = 1.5;
    sigma = (gamma(1 + beta_levy) * sin(pi * beta_levy / 2) / ...
            (gamma((1 + beta_levy) / 2) * beta_levy * 2^((beta_levy - 1) / 2)))^(1 / beta_levy); 
    
    cluster_assignments = ones(N, 1);
    cluster_best_pos = repmat(bestx, num_clusters, 1);
    
    iter = 0;
    max_iter = max(1, floor(fe_limit / N)); 
    
    while FE < fe_limit
        iter = iter + 1;
        cr = 0.6 + 0.3 * (iter / max_iter); 
        mr = 0.1 - 0.099 * (iter / max_iter); 
        
        % 1. Colony Formation
        if mod(iter, BU) == 1 || iter == 1
            warning('off', 'stats:kmeans:FailedToConverge');
            try
                [cluster_assignments, ~] = kmeans(X, num_clusters, 'MaxIter', 5, 'EmptyAction', 'singleton');
            catch
                cluster_assignments = randi(num_clusters, N, 1);
            end
            warning('on', 'stats:kmeans:FailedToConverge');
            
            for c = 1:num_clusters
                idx_c = find(cluster_assignments == c);
                if numel(idx_c) < 5 && num_clusters > 1
                    cluster_assignments(idx_c) = mode(cluster_assignments); 
                end
            end
            
            for c = 1:num_clusters
                idx_c = find(cluster_assignments == c);
                if ~isempty(idx_c)
                    [~, min_idx] = min(fit(idx_c));
                    cluster_best_pos(c, :) = X(idx_c(min_idx), :);
                end
            end
        end
        
        % 2. Food Search & Family Development
        for i = 1:N
            if FE >= fe_limit, break; end
            
            c_idx = cluster_assignments(i);
            X_best_c = cluster_best_pos(c_idx, :);
            
            r1 = rand; r2 = rand;
            theta = r1 * pi * rand(1, dim); 
            
            s_th = sin(theta);
            c_th = cos(theta);
            
            if max(s_th) > min(s_th)
                a = (s_th - min(s_th)) ./ (max(s_th) - min(s_th)); 
            else
                a = s_th;
            end
            if max(c_th) > min(c_th)
                b = (c_th - min(c_th)) ./ (max(c_th) - min(c_th)); 
            else
                b = c_th;
            end
            
            V(i, :) = w * V(i, :) + r1 .* a .* (bestx - X(i, :)) + r2 .* b .* (X_best_c - X(i, :)); 
            
            if rand < 0.5
                step = (0.77 * randn(1, dim)) .* bestx; 
                X_new = X(i, :) + step;
            else
                u = randn(1, dim) * sigma;
                v = randn(1, dim);
                levy_step = u ./ abs(v).^(1 / beta_levy);
                X_new = X(i, :) + 0.1 .* levy_step .* V(i, :); 
            end
            X_new = max(min(X_new, ub_vec), lb_vec);
            
            p2_idx = randi(N);
            X_child = cr * X_new + (1 - cr) * X(p2_idx, :); 
            X_child = X_child + randn(1, dim) * mr; 
            X_child = max(min(X_child, ub_vec), lb_vec);
            
            f_child = fobj(X_child);
            FE = FE + 1;
            
            if f_child < fit(i)
                X(i, :) = X_child;
                fit(i) = f_child;
                if f_child < bestf
                    bestf = f_child;
                    bestx = X_child;
                end
            end
            ConvergenceCurve(FE) = bestf;
        end
    end
    ConvergenceCurve = ConvergenceCurve(1:FE);
end

% --- ACSBOA Budgeted ---
function [bestx, bestf, ConvergenceCurve, FE] = ACSBOA_Budgeted(dim, lb, ub, fe_limit, N, fobj)
    FE = 0;
    if isscalar(lb), lb_vec = lb * ones(1, dim); else, lb_vec = lb(:)'; end
    if isscalar(ub), ub_vec = ub * ones(1, dim); else, ub_vec = ub(:)'; end
    
    ConvergenceCurve = zeros(1, fe_limit);
    X = repmat(lb_vec, N, 1) + rand(N, dim) .* repmat(ub_vec - lb_vec, N, 1);
    fit = inf(1, N);
    
    bestf = inf;
    bestx = X(1, :);
    
    for i = 1:N
        if FE >= fe_limit, break; end
        X(i, :) = max(min(X(i, :), ub_vec), lb_vec);
        fit(i) = fobj(X(i, :));
        FE = FE + 1;
        
        if fit(i) < bestf
            bestf = fit(i);
            bestx = X(i, :);
        end
        ConvergenceCurve(FE) = bestf;
    end
    
    MaxIter = max(1, floor((fe_limit - N) / (2 * N))); 
    t = 0;
    
    while FE < fe_limit
        t = t + 1;
        ratio = min(t / MaxIter, 1); 
        CF = exp(-3 * (ratio)^2);
        
        % 1. Predation Strategy
        for i = 1:N
            if FE >= fe_limit, break; end
            
            if ratio < 1/3
                X_random_1 = randi(N);
                X_random_2 = randi(N);
                while X_random_2 == X_random_1 && N > 1
                    X_random_2 = randi(N);
                end
                R1 = rand(1, dim);
                X1 = X(i, :) + (X(X_random_1, :) - X(X_random_2, :)) .* R1;
                
            elseif ratio >= 1/3 && ratio < 2/3
                RB = randn(1, dim);
                direction_factor = exp((ratio)^4) * (RB - 0.5);
                X1 = bestx + direction_factor .* (bestx - X(i, :));
                
            else
                if rand < 0.5
                    beta_val = 1.5 + 0.5 * ratio;
                    RL = 0.5 * ACSBOA_Levy(dim, beta_val);
                    X1 = bestx + CF * X(i, :) .* RL;
                else
                    [~, sorted_idx] = sort(fit);
                    elite_count = max(1, ceil(N / 4));
                    elite_idx = sorted_idx(1:elite_count);
                    partner_idx = elite_idx(randi(elite_count));
                    
                    alpha_vec = rand(1, dim);
                    X1 = bestx + alpha_vec .* (X(partner_idx, :) - X(i, :)) + ...
                        0.1 * randn(1, dim) .* (ub_vec - lb_vec) * (1 - ratio);
                end
            end
            
            X1 = max(min(X1, ub_vec), lb_vec);
            f_new1 = fobj(X1);
            FE = FE + 1;
            
            if f_new1 <= fit(i)
                X(i, :) = X1;
                fit(i) = f_new1;
                if f_new1 < bestf
                    bestf = f_new1;
                    bestx = X1;
                end
            end
            ConvergenceCurve(FE) = bestf;
        end
        
        if FE >= fe_limit, break; end
        
        % 2. Escape Strategy
        [~, idx] = sort(fit);
        rank = zeros(1, N);
        rank(idx) = 1:N;
        top_count = max(1, ceil(0.3 * N));
        mid_count = max(top_count, ceil(0.7 * N));
        
        for i = 1:N
            if FE >= fe_limit, break; end
            
            if rank(i) <= top_count
                RB = rand(1, dim);
                X2 = bestx + (1 - ratio)^2 * (2 * RB - 1) .* X(i, :);
                
            elseif rank(i) <= mid_count
                k_idx = randperm(N, 1);
                Xrandom = X(k_idx, :);
                R2 = rand(1, dim);
                K_val = round(1 + rand);
                X2 = X(i, :) + R2 .* (Xrandom - K_val * X(i, :));
                
            else
                best_idx = idx(randi(top_count));
                R3 = rand(1, dim);
                X2 = X(i, :) + R3 .* (X(best_idx, :) - X(i, :)) + ...
                    0.05 * ACSBOA_Levy(dim, 1.5) .* (ub_vec - lb_vec) * (1 - ratio);
            end
            
            X2 = max(min(X2, ub_vec), lb_vec);
            f_new2 = fobj(X2);
            FE = FE + 1;
            
            if f_new2 <= fit(i)
                X(i, :) = X2;
                fit(i) = f_new2;
                if f_new2 < bestf
                    bestf = f_new2;
                    bestx = X2;
                end
            end
            ConvergenceCurve(FE) = bestf;
        end
    end
    ConvergenceCurve = ConvergenceCurve(1:FE);
end

% --- ACSBOA Levy Flight Helper ---
function o = ACSBOA_Levy(d, beta)
    if nargin < 2
        beta = 1.5;
    end
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
        (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    o = u ./ abs(v).^(1 / beta);
end