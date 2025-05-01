import numpy as np

def x_from_grid(x_grid):
    x_edges = np.cumsum(
        np.insert(x_grid, 0, -2*x_grid[0])
    )
    x_centers = (
        ( x_edges + np.roll(x_edges, -1) )/2
    )[:-1]
    
    return x_centers, x_edges

def numerical_laplacian(c, dx):
    """
    Numerical lapplacian for the concentration array.
    explained in example_notebooks/diffusion_equation/1_solving_diffusion_equation
    Parameters:
    c  : np.array - concentration array
    dx : float   - spatial discretization size
    Returns:
    np.array
    """
    c_iminus  = np.roll(c, 1)
    c_iplus   = np.roll(c, -1)
    laplacian = (c_iplus - 2*c + c_iminus)/dx**2
    return laplacian
    
def laplacian_non_unif_grid(c, xcenters):
    xcenters_left = np.roll(xcenters, 1)
    xcenters_right = np.roll(xcenters, -1)
    c_left = np.roll(c, 1)
    c_right = np.roll(c, -1)
    laplacian = 2 /(xcenters_right - xcenters_left) * ( ( c_right - c )/( xcenters_right - xcenters ) \
       - ( c - c_left )/( xcenters - xcenters_left ) ) 
    return laplacian

def refl_bound(c,x_grid_prev_grid):
    c[2]  += c[1]*np.sqrt(x_grid_prev_grid[1]/x_grid_prev_grid[2])
    c[-3] += c[-2]*np.sqrt(x_grid_prev_grid[-2]/x_grid_prev_grid[-3])
    c[0], c[1], c[-1], c[-2] = 0, 0, 0, 0
    return c

def cut_edges(old_list):
    nrm = 2
    new_list = [i[nrm:-nrm] for i in old_list]
    return np.array(new_list)