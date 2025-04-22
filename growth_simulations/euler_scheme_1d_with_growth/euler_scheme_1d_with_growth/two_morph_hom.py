import numpy as np
import math
from tqdm import tqdm
import os
from scipy.optimize import fsolve

from euler_scheme_1d_with_growth.analytical_solutions import steady_solution, nrect, nheaviside, hill
from euler_scheme_1d_with_growth.functions_for_numerics import *
# GROWTH RULE
from euler_scheme_1d_with_growth.growth_rules import two_morph_growth_rule as growth_rule

def threshold_vs_Lfinal(Lfinal, x0, lam, alpha, w, beta, D):
    if Lfinal > x0 + w:
        return steady_solution( Lfinal/2, x0, lam, alpha, w, beta, D, Lfinal )
    else:
        return np.nan

def Lfinal(threshold, x0, lam, alpha, w, beta, D):   
    def equation_to_solve(Lfinal):
        return steady_solution(Lfinal/2, x0, lam, alpha, w, beta, D, Lfinal) - threshold
    solution = fsolve(equation_to_solve, 2*(x0+w) )[0]
    return solution

class two_morph_euler:
    def __init__(self, D, beta, alpha, L0, x0, w, g, threshold_growth, ndx, dx, tmax, dt, t, ndt, ndt_to_save):
        # model parm
        self.D = D
        self.beta = beta
        self.alpha = alpha
        self.lam = math.sqrt(D / beta)
        self.L0 = L0
        self.x0 = x0
        self.w  = w
        self.g  = g
        self.threshold_growth = threshold_growth


        # simulation parm
        self.small_number = 1e-6
        self.ndx = ndx + 4 # +4 for REFLECTING boundary conditions
        self.dx  = dx
        self.tmax = tmax
        self.dt  = dt
        self.t   = t
        self.ndt = ndt
        self.ndt_to_save = ndt_to_save

        # Initialize arrays
        self.initialize_arrays()
    
    def source_s( self, x, x0, w, L ):
        source_shape = self.alpha * nrect(x, x0, w, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def source_f( self, x, x0, w, L ):
        source_shape = self.alpha * nrect(x, L - (w + x0), w, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def initialize_arrays(self):

        # save NOT all time points, to increase speed and save space
        self.n_to_save = np.linspace(0, self.ndt - 1, self.ndt_to_save).astype(int)
        self.time_to_save = self.dt * self.n_to_save

        # Initialize grids, concentration, sources, growth, and saved values
        self.x_gridt = np.zeros((self.ndt_to_save, self.ndx))
        self.x_centerst = np.zeros((self.ndt_to_save, self.ndx))
        self.x_edgest = np.zeros((self.ndt_to_save, self.ndx + 1))

        self.s_steadyt = np.zeros((self.ndt_to_save, self.ndx))
        self.f_steadyt = np.zeros((self.ndt_to_save, self.ndx))
        self.st = np.zeros((self.ndt_to_save, self.ndx))
        self.ft = np.zeros((self.ndt_to_save, self.ndx))

        self.source_st = np.zeros((self.ndt_to_save, self.ndx))
        self.source_ft = np.zeros((self.ndt_to_save, self.ndx))
        self.grt = np.zeros((self.ndt_to_save, self.ndx))

        self.Lt = np.zeros(self.ndt_to_save)
        self.grsizet = np.zeros(self.ndt_to_save)

        # Set initial conditions
        self.x_gridt[0] = self.dx * np.ones(self.ndx)
        self.x_centers, self.x_edges = x_from_grid(self.x_gridt[0])
        self.x_centerst[0] = self.x_centers
        self.x_edgest[0] = self.x_edges

        self.s_steady = steady_solution(self.x_centerst[0], self.x0, self.lam, self.alpha, self.w, self.beta, self.D, self.L0)
        self.f_steady = steady_solution(self.x_centerst[0], self.L0 - self.w - self.x0, self.lam, self.alpha, self.w, self.beta, self.D, self.L0)
        self.s_steadyt[0] = self.s_steady
        self.f_steadyt[0] = self.f_steady
        self.st[0] = self.s_steady
        self.ft[0] = self.f_steady

        self.source_st[0] = self.source_s(self.x_centerst[0], self.x0, self.w, self.L0)
        self.source_ft[0] = self.source_f(self.x_centerst[0], self.x0, self.w, self.L0)
        self.grt[0] = growth_rule(self.st[0], self.ft[0], self.threshold_growth, self.small_number)

        self.Lt[0] = np.sum(self.x_gridt[0][2:-2])
        self.grsizet[0] = np.sum(self.x_gridt[0][np.nonzero(self.grt[0])[0]])

    def run_simulation(self):
        x_grid_prev = self.dx * np.ones(self.ndx)
        s_prev = self.s_steady.copy()
        f_prev = self.f_steady.copy()
        save_index = 1

        # flag showing that growth has stopped, while for the assumed models of growth has stopped it does not recover
        # NEEDS TO BE RECONSIDERED FOR POSITIVE FEEDBACK
        growth_has_stopped = False 

        for n in tqdm(range(1, self.ndt)):

            if growth_has_stopped == False:
                # growth
                growth_region = growth_rule( s_prev, f_prev, self.threshold_growth, self.small_number) 
                # ideally, would need to re-mesh for an increasing bin size
                
                # instead stop growth, if growth region is down to less than 5 bins in size
                if np.sum(growth_region) < 3:
                    growth_region = 0 * growth_region
                    growth_has_stopped = True

            x_grid_prev_gr = x_grid_prev * (1 + self.g * self.dt * growth_region )
            x_centers_prev_gr, x_edges_prev_gr = x_from_grid(x_grid_prev_gr)
            L = x_edges_prev_gr[-3]

            # Euler scheme for concentration
            s_prev = s_prev * (x_grid_prev / x_grid_prev_gr)
            f_prev = f_prev * (x_grid_prev / x_grid_prev_gr)

            sn = self.D * laplacian_non_unif_grid(s_prev, x_centers_prev_gr) * self.dt + s_prev - self.beta * s_prev * self.dt
            fn = self.D * laplacian_non_unif_grid(f_prev, x_centers_prev_gr) * self.dt + f_prev - self.beta * f_prev * self.dt

            sn += self.dt * self.source_s(x_centers_prev_gr, self.x0, self.w, L)
            fn += self.dt * self.source_f(x_centers_prev_gr, self.x0, self.w, L)

            sn = refl_bound(sn)
            fn = refl_bound(fn)

            # Steady state at this L(t)
            s_steady = steady_solution(x_centers_prev_gr, self.x0, self.lam, self.alpha, self.w, self.beta, self.D, L)
            f_steady = steady_solution(x_centers_prev_gr, L - self.w - self.x0, self.lam, self.alpha, self.w, self.beta, self.D, L)

            # Save for next time step
            x_grid_prev = x_grid_prev_gr
            s_prev = sn
            f_prev = fn

            # Update corresponding arrays
            if n in self.n_to_save:
                self.x_gridt[save_index] = x_grid_prev_gr
                self.x_centerst[save_index] = x_centers_prev_gr
                self.x_edgest[save_index] = x_edges_prev_gr

                self.s_steadyt[save_index] = s_steady
                self.f_steadyt[save_index] = f_steady
                self.st[save_index] = sn
                self.ft[save_index] = fn

                self.source_st[save_index] = self.source_s(x_centers_prev_gr, self.x0, self.w, L)
                self.source_ft[save_index] = self.source_f(x_centers_prev_gr, self.x0, self.w, L)

                self.grt[save_index] = growth_region
                self.Lt[save_index] = L
                self.grsizet[save_index] = np.sum(x_grid_prev_gr[np.argwhere( self.grt[save_index] == 1 )]) if np.sum(self.grt[save_index]) > 0 else 0
                save_index += 1

        self.post_process()


    def post_process(self):
        # Cut edges for final results
        self.x_edgest, self.x_centerst, self.x_gridt = cut_edges(self.x_edgest), cut_edges(self.x_centerst), cut_edges(self.x_gridt)
        self.s_steadyt, self.f_steadyt = cut_edges(self.s_steadyt), cut_edges(self.f_steadyt)
        self.st, self.ft = cut_edges(self.st), cut_edges(self.ft)
        self.source_st, self.source_ft = cut_edges(self.source_st), cut_edges(self.source_ft)
        self.grt = cut_edges(self.grt)

        # Amplitude of morphogen profiles
        self.At = np.max(self.s_steadyt, axis=1)

    def save_results(self, folder_base='results_of_numerics/two_morph'):
        # Create folder path
        folder_path = os.path.join(folder_base, f'th_{self.threshold_growth}_g_{self.g}')

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at: {folder_path}")
        else:
            print(f"Folder already exists at: {folder_path}")

        # Save time-related data
        np.save(os.path.join(folder_path, 'time_to_save.npy'), self.time_to_save)
        np.save(os.path.join(folder_path, 'x_centerst.npy'), self.x_centerst)
        np.save(os.path.join(folder_path, 'x_gridt.npy'),    self.x_gridt)

        # Save concentration profiles
        np.save(os.path.join(folder_path, 's_steadyt.npy'), self.s_steadyt)
        np.save(os.path.join(folder_path, 'f_steadyt.npy'), self.f_steadyt)
        np.save(os.path.join(folder_path, 'st.npy'), self.st)
        np.save(os.path.join(folder_path, 'ft.npy'), self.ft)

        # Save source profiles
        np.save(os.path.join(folder_path, 'source_st.npy'), self.source_st)
        np.save(os.path.join(folder_path, 'source_ft.npy'), self.source_ft)

        # Save growth data
        np.save(os.path.join(folder_path, 'grt.npy'), self.grt)
        np.save(os.path.join(folder_path, 'grsizet.npy'), self.grsizet)
        np.save(os.path.join(folder_path, 'Lt.npy'), self.Lt)

        print(f"Results successfully saved in: {folder_path}")


class two_morph_euler_steady:
    def __init__(self, D, beta, alpha, L0, x0, w, g, threshold_growth, ndx, dx, tmax, dt, t, ndt, ndt_to_save):
        # model parm
        self.D = D
        self.beta = beta
        self.alpha = alpha
        self.lam = math.sqrt(D / beta)
        self.L0 = L0
        self.x0 = x0
        self.w  = w
        self.g  = g
        self.threshold_growth = threshold_growth


        # simulation parm
        self.small_number = 1e-6
        self.ndx = ndx + 4 # +4 for REFLECTING boundary conditions
        self.dx  = dx
        self.tmax = tmax
        self.dt  = dt
        self.t   = t
        self.ndt = ndt
        self.ndt_to_save = ndt_to_save

        # Initialize arrays
        self.initialize_arrays()
    
    def source_s( self, x, x0, w, L ):
        source_shape = self.alpha * nrect(x, x0, w, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def source_f( self, x, x0, w, L ):
        source_shape = self.alpha * nrect(x, L - (w + x0), w, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def initialize_arrays(self):

        # save NOT all time points, to increase speed and save space
        self.n_to_save = np.linspace(0, self.ndt - 1, self.ndt_to_save).astype(int)
        self.time_to_save = self.dt * self.n_to_save

        # Initialize grids, concentration, sources, growth, and saved values
        self.x_gridt = np.zeros((self.ndt_to_save, self.ndx))
        self.x_centerst = np.zeros((self.ndt_to_save, self.ndx))
        self.x_edgest = np.zeros((self.ndt_to_save, self.ndx + 1))

        self.s_steadyt = np.zeros((self.ndt_to_save, self.ndx))
        self.f_steadyt = np.zeros((self.ndt_to_save, self.ndx))

        self.source_st = np.zeros((self.ndt_to_save, self.ndx))
        self.source_ft = np.zeros((self.ndt_to_save, self.ndx))
        self.grt = np.zeros((self.ndt_to_save, self.ndx))

        self.Lt = np.zeros(self.ndt_to_save)
        self.grsizet = np.zeros(self.ndt_to_save)

        # Set initial conditions
        self.x_gridt[0] = self.dx * np.ones(self.ndx)
        self.x_centers, self.x_edges = x_from_grid(self.x_gridt[0])
        self.x_centerst[0] = self.x_centers
        self.x_edgest[0] = self.x_edges

        self.s_steady = steady_solution(self.x_centerst[0], self.x0, self.lam, self.alpha, self.w, self.beta, self.D, self.L0)
        self.f_steady = steady_solution(self.x_centerst[0], self.L0 - self.w - self.x0, self.lam, self.alpha, self.w, self.beta, self.D, self.L0)
        self.s_steadyt[0] = self.s_steady
        self.f_steadyt[0] = self.f_steady

        self.source_st[0] = self.source_s(self.x_centerst[0], self.x0, self.w, self.L0)
        self.source_ft[0] = self.source_f(self.x_centerst[0], self.x0, self.w, self.L0)
        self.grt[0] = growth_rule(self.s_steadyt[0], self.f_steadyt[0], self.threshold_growth, self.small_number)

        self.Lt[0] = np.sum(self.x_gridt[0][2:-2])
        self.grsizet[0] = np.sum(self.x_gridt[0][np.nonzero(self.grt[0])[0]])

    def run_simulation(self):
        x_grid_prev = self.dx * np.ones(self.ndx)
        s_prev = self.s_steady.copy()
        f_prev = self.f_steady.copy()
        save_index = 1

        # flag showing that growth has stopped, while for the assumed models of growth has stopped it does not recover
        # NEEDS TO BE RECONSIDERED FOR POSITIVE FEEDBACK
        growth_has_stopped = False 

        for n in tqdm(range(1, self.ndt)):

            if growth_has_stopped == False:
                # growth
                growth_region = growth_rule( s_prev, f_prev, self.threshold_growth, self.small_number) 
                # ideally, would need to re-mesh for an increasing bin size
                
                # instead stop growth, if growth region is down to less than 5 bins in size
                if np.sum(growth_region) < 3:
                    growth_region = 0 * growth_region
                    growth_has_stopped = True

            x_grid_prev_gr = x_grid_prev * (1 + self.g * self.dt * growth_region )
            x_centers_prev_gr, x_edges_prev_gr = x_from_grid(x_grid_prev_gr)
            L = x_edges_prev_gr[-3]

            # Steady state at this L(t)
            s_steady = steady_solution(x_centers_prev_gr, self.x0, self.lam, self.alpha, self.w, self.beta, self.D, L)
            f_steady = steady_solution(x_centers_prev_gr, L - self.w - self.x0, self.lam, self.alpha, self.w, self.beta, self.D, L)

            # Save for next time step
            x_grid_prev = x_grid_prev_gr
            s_prev = s_steady
            f_prev = f_steady

            # Update corresponding arrays
            if n in self.n_to_save:
                self.x_gridt[save_index] = x_grid_prev_gr
                self.x_centerst[save_index] = x_centers_prev_gr
                self.x_edgest[save_index] = x_edges_prev_gr

                self.s_steadyt[save_index] = s_steady
                self.f_steadyt[save_index] = f_steady

                self.source_st[save_index] = self.source_s(x_centers_prev_gr, self.x0, self.w, L)
                self.source_ft[save_index] = self.source_f(x_centers_prev_gr, self.x0, self.w, L)

                self.grt[save_index] = growth_region
                self.Lt[save_index] = L
                self.grsizet[save_index] = np.sum(x_grid_prev_gr[np.argwhere( self.grt[save_index] == 1 )]) if np.sum(self.grt[save_index]) > 0 else 0
                save_index += 1

        self.post_process()


    def post_process(self):
        # Cut edges for final results
        self.x_edgest, self.x_centerst, self.x_gridt = cut_edges(self.x_edgest), cut_edges(self.x_centerst), cut_edges(self.x_gridt)
        self.s_steadyt, self.f_steadyt = cut_edges(self.s_steadyt), cut_edges(self.f_steadyt)
        self.source_st, self.source_ft = cut_edges(self.source_st), cut_edges(self.source_ft)
        self.grt = cut_edges(self.grt)

        # Amplitude of morphogen profiles
        self.At = np.max(self.s_steadyt, axis=1)

    def save_results(self, folder_base='results_of_numerics/two_morph'):
        # Create folder path
        folder_path = os.path.join(folder_base, f'th_{self.threshold_growth}_g_{self.g}')

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at: {folder_path}")
        else:
            print(f"Folder already exists at: {folder_path}")

        # Save time-related data
        np.save(os.path.join(folder_path, 'time_to_save.npy'), self.time_to_save)
        np.save(os.path.join(folder_path, 'x_centerst.npy'), self.x_centerst)
        np.save(os.path.join(folder_path, 'x_gridt.npy'),    self.x_gridt)

        # Save concentration profiles
        np.save(os.path.join(folder_path, 's_steadyt.npy'), self.s_steadyt)
        np.save(os.path.join(folder_path, 'f_steadyt.npy'), self.f_steadyt)

        # Save source profiles
        np.save(os.path.join(folder_path, 'source_st.npy'), self.source_st)
        np.save(os.path.join(folder_path, 'source_ft.npy'), self.source_ft)

        # Save growth data
        np.save(os.path.join(folder_path, 'grt.npy'), self.grt)
        np.save(os.path.join(folder_path, 'grsizet.npy'), self.grsizet)
        np.save(os.path.join(folder_path, 'Lt.npy'), self.Lt)

        print(f"Results successfully saved in: {folder_path}")


class two_morph_feedback_euler_steady:
    def __init__(self, D, beta, alpha, L0, wComp, g, threshold_growth, threshold_fb, ndx, dx, tmax, dt, t, ndt, ndt_to_save):
        # model parm
        self.D = D
        self.beta = beta
        self.alpha = alpha
        self.lam = math.sqrt(D / beta)
        self.L0 = L0
        self.wComp = wComp
        self.g  = g
        self.threshold_growth = threshold_growth
        self.threshold_fb = threshold_fb

        # simulation parm
        self.small_number = 1e-12
        self.ndx = ndx + 4 # +4 for REFLECTING boundary conditions
        self.dx  = dx
        self.tmax = tmax
        self.dt  = dt
        self.t   = t
        self.ndt = ndt
        self.ndt_to_save = ndt_to_save

        # Initialize arrays
        self.initialize_arrays()
    
    def source_s( self, x, f, L ):
        source_shape = self.alpha * nheaviside(self.wComp - x, self.small_number) * nheaviside(f - self.threshold_fb, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def source_f( self, x, s, L ):
        source_shape = self.alpha * nheaviside( x - (L - self.wComp), self.small_number) * nheaviside(s - self.threshold_fb, self.small_number)
        source_shape[0] = source_shape[1] = source_shape[-1] = source_shape[-2] = 0
        return source_shape

    def initialize_arrays(self):

        # save NOT all time points, to increase speed and save space
        self.n_to_save = np.linspace(0, self.ndt - 1, self.ndt_to_save).astype(int)
        self.time_to_save = self.dt * self.n_to_save

        # Initialize grids, concentration, sources, growth, and saved values
        self.x_gridt = np.zeros((self.ndt_to_save, self.ndx))
        self.x_centerst = np.zeros((self.ndt_to_save, self.ndx))
        self.x_edgest = np.zeros((self.ndt_to_save, self.ndx + 1))

        self.s_steadyt = np.zeros((self.ndt_to_save, self.ndx))
        self.f_steadyt = np.zeros((self.ndt_to_save, self.ndx))

        self.source_st = np.zeros((self.ndt_to_save, self.ndx))
        self.source_ft = np.zeros((self.ndt_to_save, self.ndx))
        self.grt = np.zeros((self.ndt_to_save, self.ndx))

        self.Lt = np.zeros(self.ndt_to_save)
        self.grsizet = np.zeros(self.ndt_to_save)

        # Set initial conditions
        self.x_gridt[0] = self.dx * np.ones(self.ndx)
        self.x_centers, self.x_edges = x_from_grid(self.x_gridt[0])
        self.x_centerst[0] = self.x_centers
        self.x_edgest[0] = self.x_edges



        s_steady = steady_solution( self.x_centerst[0], 0, self.lam, self.alpha, self.wComp, self.beta, self.D, self.L0) 
        if np.any( (s_steady < self.threshold_fb) * (self.x_centerst[0] < self.wComp)):
            def equation_to_solve(x0):
                return steady_solution(self.L0 - x0, x0, self.lam, self.alpha, self.wComp - x0, self.beta, self.D, self.L0) - self.threshold_fb
            x0 = fsolve(equation_to_solve, 0)[0]
        else:
            x0 = 0

        self.s_steady = steady_solution(self.x_centerst[0], x0, self.lam, self.alpha, self.wComp - x0, self.beta, self.D, self.L0)
        self.f_steady = steady_solution(self.x_centerst[0], self.L0 - self.wComp, self.lam, self.alpha, self.wComp - x0, self.beta, self.D, self.L0)

        self.s_steadyt[0] = self.s_steady
        self.f_steadyt[0] = self.f_steady

        self.source_st[0] = self.source_s(self.x_centerst[0], self.f_steadyt[0], self.L0)
        self.source_ft[0] = self.source_f(self.x_centerst[0], self.s_steadyt[0], self.L0)

        self.grt[0] = growth_rule(self.s_steadyt[0], self.f_steadyt[0], self.threshold_growth, self.small_number)

        self.Lt[0] = np.sum(self.x_gridt[0][2:-2])
        self.grsizet[0] = np.sum(self.x_gridt[0][np.nonzero(self.grt[0])[0]])

    def run_simulation(self):
        x_grid_prev = self.dx * np.ones(self.ndx)
        s_prev = self.s_steady.copy()
        f_prev = self.f_steady.copy()
        save_index = 1

        # flag showing that growth has stopped, while for the assumed models of growth has stopped it does not recover
        # NEEDS TO BE RECONSIDERED FOR POSITIVE FEEDBACK
        growth_has_stopped = False 

        for n in tqdm(range(1, self.ndt)):

            if growth_has_stopped == False:
                # growth
                growth_region = growth_rule( s_prev, f_prev, self.threshold_growth, self.small_number) 
                # ideally, would need to re-mesh for an increasing bin size
                
                # instead stop growth, if growth region is down to less than 5 bins in size
                if np.sum(growth_region) < 3:
                    growth_region = 0 * growth_region
                    growth_has_stopped = True

            x_grid_prev_gr = x_grid_prev * (1 + self.g * self.dt * growth_region )
            x_centers_prev_gr, x_edges_prev_gr = x_from_grid(x_grid_prev_gr)
            L = x_edges_prev_gr[-3]

            # source regions
            source_st = self.source_s(x_centers_prev_gr, f_prev, L)
            source_ft = self.source_f(x_centers_prev_gr, s_prev, L)

            if np.sum( source_st ) != 0:
                x0 = x_edges_prev_gr[ np.nonzero( source_st )[0][0] ]
                # Steady state at this L(t)
                s_steady = steady_solution(x_centers_prev_gr, x0, self.lam, self.alpha, self.wComp - x0, self.beta, self.D, L)
                f_steady = steady_solution(x_centers_prev_gr, L - self.wComp, self.lam, self.alpha, self.wComp - x0, self.beta, self.D, L)
            else:
                s_steady = np.zeros_like(x_centers_prev_gr)
                f_steady = np.zeros_like(x_centers_prev_gr)

            # Save for next time step
            x_grid_prev = x_grid_prev_gr
            s_prev = s_steady
            f_prev = f_steady

            # Update corresponding arrays
            if n in self.n_to_save:
                self.x_gridt[save_index] = x_grid_prev_gr
                self.x_centerst[save_index] = x_centers_prev_gr
                self.x_edgest[save_index] = x_edges_prev_gr

                self.s_steadyt[save_index] = s_steady
                self.f_steadyt[save_index] = f_steady

                self.source_st[save_index] = source_st
                self.source_ft[save_index] = source_ft

                self.grt[save_index] = growth_region
                self.Lt[save_index] = L
                self.grsizet[save_index] = np.sum(x_grid_prev_gr[np.argwhere( self.grt[save_index] == 1 )]) if np.sum(self.grt[save_index]) > 0 else 0
                save_index += 1

        self.post_process()


    def post_process(self):
        # Cut edges for final results
        self.x_edgest, self.x_centerst, self.x_gridt = cut_edges(self.x_edgest), cut_edges(self.x_centerst), cut_edges(self.x_gridt)
        self.s_steadyt, self.f_steadyt = cut_edges(self.s_steadyt), cut_edges(self.f_steadyt)
        self.source_st, self.source_ft = cut_edges(self.source_st), cut_edges(self.source_ft)
        self.grt = cut_edges(self.grt)

        # Amplitude of morphogen profiles
        self.At = np.max(self.s_steadyt, axis=1)

    def save_results(self, folder_base='results_of_numerics/two_morph'):
        # Create folder path
        folder_path = os.path.join(folder_base, f'th_{self.threshold_growth}_g_{self.g}')

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at: {folder_path}")
        else:
            print(f"Folder already exists at: {folder_path}")

        # Save time-related data
        np.save(os.path.join(folder_path, 'time_to_save.npy'), self.time_to_save)
        np.save(os.path.join(folder_path, 'x_centerst.npy'), self.x_centerst)
        np.save(os.path.join(folder_path, 'x_gridt.npy'),    self.x_gridt)

        # Save concentration profiles
        np.save(os.path.join(folder_path, 's_steadyt.npy'), self.s_steadyt)
        np.save(os.path.join(folder_path, 'f_steadyt.npy'), self.f_steadyt)

        # Save source profiles
        np.save(os.path.join(folder_path, 'source_st.npy'), self.source_st)
        np.save(os.path.join(folder_path, 'source_ft.npy'), self.source_ft)

        # Save growth data
        np.save(os.path.join(folder_path, 'grt.npy'), self.grt)
        np.save(os.path.join(folder_path, 'grsizet.npy'), self.grsizet)
        np.save(os.path.join(folder_path, 'Lt.npy'), self.Lt)

        print(f"Results successfully saved in: {folder_path}")