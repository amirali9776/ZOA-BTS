import numpy as np

def buzjani_local_search(bestx, bestf, fe_budget, fobj, lb_vec, ub_vec):
    """
    Buzjani Tiling Strategy (BTS) - Local Search Operator
    
    Inputs:
        bestx     - 1D numpy array: The best solution found by the base algorithm
        bestf     - Float: The objective value of bestx
        fe_budget - Int: Maximum number of evaluations allowed for BTS
        fobj      - Callable: The objective/cost function fobj(x)
        lb_vec    - 1D numpy array: Lower bounds of the search space
        ub_vec    - 1D numpy array: Upper bounds of the search space
        
    Outputs:
        bestx     - 1D numpy array: The refined optimal solution
        bestf     - Float: The refined optimal objective value
        curve     - 1D numpy array: The convergence history during BTS
        FE        - Int: Total FEs consumed (will equal fe_budget)
    """
    
    FE = 0
    curve = np.full(fe_budget, np.inf)  # Preallocate convergence curve
    
    # Ensure inputs are numpy arrays for vector operations
    bestx = np.array(bestx, dtype=float)
    lb_vec = np.array(lb_vec, dtype=float)
    ub_vec = np.array(ub_vec, dtype=float)
    
    dim = len(bestx)
    Delta = np.mean(ub_vec - lb_vec) # Mean parameter space scale
    
    # --- Control Parameters ---
    omega = 0.10   # Scaling factor for the initial search radius
    eps = np.finfo(float).eps # Machine epsilon to prevent division by zero
    
    # --- Main Exploitation Loop ---
    while FE < fe_budget:
        
        # 1. Dynamic Radius Computation (Shrinking Trust Region)
        progress = FE / fe_budget 
        radius = (Delta * omega * (1 - progress)**3) + 1e-12 
        
        base_pos = np.copy(bestx)
        base_fit = bestf
        
        # 2. Subspace Selection (k = 2)
        if dim > 1:
            # Randomly select 2 distinct dimensions (0-based)
            dims_selected = np.random.choice(dim, 2, replace=False)
            k_dims = 2
        else:
            dims_selected = [0]
            k_dims = 1
            
        improved = False
        
        # =================================================================
        # Phase A: Hexagonal Tiling (2D Subspace)
        # =================================================================
        if k_dims == 2:
            
            # Construct Orthonormal Basis (Gram-Schmidt process)
            r1 = np.random.randn(2)
            r2 = np.random.randn(2)
            
            u = r1 / (np.linalg.norm(r1) + eps)
            v_raw = r2 - (np.dot(r2, u) * u)
            v = v_raw / (np.linalg.norm(v_raw) + eps)
            
            # Generate base phase shift for hexagonal rotation
            phi = 2 * np.pi * np.random.rand()
            hex_angles = np.arange(6) * (np.pi / 3)
            
            best_hex_fit = base_fit
            best_hex_pos = np.copy(base_pos)
            
            # Evaluate the 6 vertices of the hexagon
            for a in hex_angles:
                if FE >= fe_budget:
                    break
                
                ang = a + phi
                
                # Projection vector mapped to the 2D plane
                offset_2d = radius * (np.cos(ang)*u + np.sin(ang)*v)
                
                cand = np.copy(base_pos)
                cand[dims_selected[0]] += offset_2d[0]
                cand[dims_selected[1]] += offset_2d[1]
                
                # Standard Boundary Clamp using numpy clip
                cand = np.clip(cand, lb_vec, ub_vec)
                
                # Evaluation
                f_cand = fobj(cand)
                FE += 1
                
                # Greedy update for the hexagon phase
                if f_cand < best_hex_fit:
                    best_hex_fit = f_cand
                    best_hex_pos = np.copy(cand)
                
                # Log the best-so-far globally (0-based indexing)
                curve[FE - 1] = min(bestf, best_hex_fit)
                
            # Accept hexagonal improvement globally
            if best_hex_fit < bestf:
                bestf = best_hex_fit
                bestx = np.copy(best_hex_pos)
                improved = True
                
            # =============================================================
            # Phase B: Fallback Stochastic Axis-Aligned Step
            # =============================================================
            # If the hexagon failed, re-probe along the major axes.
            if not improved:
                best_ax_fit = bestf
                best_ax_pos = np.copy(bestx)
                
                for d in dims_selected:
                    for s in [-1, 1]: # Probe both positive and negative directions
                        if FE >= fe_budget:
                            break
                        
                        rho = np.random.rand() # Stochastic magnitude
                        cand = np.copy(bestx)
                        cand[d] += (s * rho * radius)
                        
                        cand = np.clip(cand, lb_vec, ub_vec)
                        f_cand = fobj(cand)
                        FE += 1
                        
                        if f_cand < best_ax_fit:
                            best_ax_fit = f_cand
                            best_ax_pos = np.copy(cand)
                        
                        curve[FE - 1] = min(bestf, best_ax_fit)
                        
                    if FE >= fe_budget:
                        break
                        
                # Accept axis fallback improvement globally
                if best_ax_fit < bestf:
                    bestf = best_ax_fit
                    bestx = np.copy(best_ax_pos)
                    
        # =================================================================
        # Phase C: Edge Case (1-Dimensional Problem)
        # =================================================================
        else:
            best_1d_fit = bestf
            best_1d_pos = np.copy(bestx)
            
            for s in [-1, 1]:
                if FE >= fe_budget:
                    break
                
                rho = np.random.rand()
                cand = np.copy(bestx)
                cand[0] += (s * rho * radius)
                
                cand = np.clip(cand, lb_vec, ub_vec)
                f_cand = fobj(cand)
                FE += 1
                
                if f_cand < best_1d_fit:
                    best_1d_fit = f_cand
                    best_1d_pos = np.copy(cand)
                
                curve[FE - 1] = min(bestf, best_1d_fit)
                
            if best_1d_fit < bestf:
                bestf = best_1d_fit
                bestx = np.copy(best_1d_pos)

    # Trim any excess preallocated indices and return
    return bestx, bestf, curve[:FE], FE