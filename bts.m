% =========================================================================
% Buzjani Tiling Strategy (BTS) - Local Search Operator
% =========================================================================
% This function applies a lightweight, high-precision geometric local search
% around the best solution found by a base optimizer. It operates strictly
% within a defined Function Evaluation (FE) budget.
%
% Inputs:
%   bestx     - 1xD vector: The best solution found by the base algorithm
%   bestf     - Scalar: The objective value of bestx
%   fe_budget - Integer: Maximum number of evaluations allowed for BTS
%   fobj      - Function handle: The objective/cost function
%   lb_vec    - 1xD vector: Lower bounds of the search space
%   ub_vec    - 1xD vector: Upper bounds of the search space
%
% Outputs:
%   bestx     - 1xD vector: The refined optimal solution
%   bestf     - Scalar: The refined optimal objective value
%   curve     - 1xFE_budget vector: The convergence history during BTS
%   FE        - Integer: Total FEs consumed (will equal fe_budget)
% =========================================================================

function [bestx, bestf, curve, FE] = BuzjaniLocalSearch(bestx, bestf, fe_budget, fobj, lb_vec, ub_vec)
    
    FE = 0;
    curve = inf(1, fe_budget);  % Preallocate convergence curve
    dim = numel(bestx);
    Delta = mean(ub_vec - lb_vec); % Mean parameter space scale
    
    % --- Control Parameters ---
    omega = 0.10;   % Scaling factor for the initial search radius
    
    % --- Main Exploitation Loop ---
    while FE < fe_budget
        
        % 1. Dynamic Radius Computation (Shrinking Trust Region)
        % The radius decays cubically as the FE budget is consumed
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
        
        % =================================================================
        % Phase A: Hexagonal Tiling (2D Subspace)
        % =================================================================
        if k_dims == 2
            
            % Construct Orthonormal Basis (Gram-Schmidt process)
            r1 = randn(2,1); 
            r2 = randn(2,1);
            
            u = r1 / (norm(r1) + eps);
            v_raw = r2 - (dot(r2, u) * u);
            v = v_raw / (norm(v_raw) + eps);
            
            % Generate base phase shift for hexagonal rotation
            phi = 2 * pi * rand();
            hex_angles = (0:5) * (pi/3);
            
            best_hex_fit = base_fit;
            best_hex_pos = base_pos;
            
            % Evaluate the 6 vertices of the hexagon
            for a = hex_angles
                if FE >= fe_budget, break; end
                
                ang = a + phi;
                
                % Projection vector mapped to the 2D plane
                offset_2d = radius * (cos(ang)*u + sin(ang)*v);
                
                cand = base_pos;
                cand(dims_selected(1)) = cand(dims_selected(1)) + offset_2d(1);
                cand(dims_selected(2)) = cand(dims_selected(2)) + offset_2d(2);
                
                % Standard Boundary Clamp
                cand = max(min(cand, ub_vec), lb_vec);
                
                % Evaluation
                f_cand = fobj(cand);
                FE = FE + 1;
                
                % Greedy update for the hexagon phase
                if f_cand < best_hex_fit
                    best_hex_fit = f_cand;
                    best_hex_pos = cand;
                end
                
                % Log the best-so-far globally
                curve(FE) = min(bestf, best_hex_fit);
            end
            
            % Accept hexagonal improvement globally
            if best_hex_fit < bestf
                bestf = best_hex_fit;
                bestx = best_hex_pos;
                improved = true;
            end
            
            % =============================================================
            % Phase B: Fallback Stochastic Axis-Aligned Step
            % =============================================================
            % If the hexagon failed to find a better solution, the space 
            % might be highly non-convex. Re-probe along the major axes.
            if ~improved
                best_ax_fit = bestf;
                best_ax_pos = bestx;
                
                for d = dims_selected
                    for s = [-1, 1] % Probe both positive and negative directions
                        if FE >= fe_budget, break; end
                        
                        rho = rand(); % Stochastic magnitude
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
                
                % Accept axis fallback improvement globally
                if best_ax_fit < bestf
                    bestf = best_ax_fit;
                    bestx = best_ax_pos;
                end
            end
            
        % =================================================================
        % Phase C: Edge Case (1-Dimensional Problem)
        % =================================================================
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
    end % End of While Loop
    
    % Trim any excess preallocated indices
    curve = curve(1:FE);
end