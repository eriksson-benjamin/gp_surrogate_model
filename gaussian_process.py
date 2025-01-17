# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:13:06 2024

@author: benjer
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.lines as mlines
import sys
sys.path.insert(0, 'C:/Users/benjer/python/utilities')
import samplers
import plot_utils
import distribution_functions as df


def gp_prediction(l, sigma_f, sigma_n , X_train, y_train, X_test):
    """
    Apply Gaussian Process regression to fit a function to the data.

    Parameters
    ----------
    l : float,
        length scale for the RBF kernel
    sigma_f : float,
        multiplicative factor for the constant kernel
    sigma_n : float,
        additive factor for the variance in the prediction
    X_train : array_like,
        independent variable of training data
    y_train : array_like,
        dependent variable of training data
    X_test : array_like,
        independent variable of test data

    Returns
    -------
    tuple : (y_pred, gp)
        y_pred : array_like,
            predicted values for X_test
        gp : GaussianProcessRegressor object,
            fitted model
    """    

    # Kernel definition 
    kernel = (ConstantKernel(constant_value=sigma_f, 
                            constant_value_bounds=(1e-2, 1e2)) *
              RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2)))
    
    # GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, 
                                  n_restarts_optimizer=10)
        
    # Fitting in the gp model
    gp.fit(X_train, y_train)
    
    # Make the prediction on test set
    y_pred, y_std = gp.predict(X_test, return_std=True)
    return y_pred, y_std, gp


def apply_gp(d_fcn, bounds, n_samples, contour_levels, true_minima=None, log=False):
    sobol_samples = samplers.get_sobol_samples(bounds, num_samples=n_samples)
    
    # Calculate distribution function
    d_vals = d_fcn(sobol_samples['x1'], sobol_samples['x2'])
    
    # Add noise
    noise_level = 0.05
    eps = np.random.uniform(1 - noise_level, 1 + noise_level, n_samples)
    d_vals *= eps
    
    # Uncertainty in each measurement
    d_uncrt = noise_level * d_vals
    
    # Axis for Gaussian process regression
    n_train = 19600
    gp_axis = samplers.get_sobol_samples(bounds, num_samples=n_train)
    
    # Training data for GP
    x_train = np.array([sobol_samples['x1'], sobol_samples['x2']]).T
    y_train = np.copy(d_vals)
    uy_train = np.copy(d_uncrt)
    
    # Test data for GP
    x_test = np.array([gp_axis['x1'], gp_axis['x2']]).T
    
    l_init = 1
    sigma_f_init = 3
    y_pred, y_std, gp = gp_prediction(l_init, sigma_f_init, uy_train, 
                                      x_train, y_train, x_test)
    
    c1, s1 = plot_utils.plot_contour(sobol_samples['x1'], sobol_samples['x2'], 
                                     d_vals, 'Sampled distribution',
                                     100, log)
    
    c2, s2 = plot_utils.plot_contour(gp_axis['x1'], gp_axis['x2'], y_pred,
                                     'GP regression', 100, log)
    
    if true_minima:
        x1_min = true_minima[0]
        x2_min = true_minima[1]
        for ax in [c1.axes, s1.axes, c2.axes, s2.axes]:
            ax.plot(x1_min, x2_min, 'rx', linestyle='None')
    
    # Comparison between GP predicted values and actual values
    y_true = d_fcn(gp_axis['x1'], gp_axis['x2'])
    plt.figure('Residuals')
    ax1 = plt.gca()
    ax1.hist((y_true - y_pred) / y_true, bins=np.arange(-1, 1, 0.01))
    ax1.axvline(-0.05, color='r', linestyle='--')
    ax1.axvline(0.05, color='r', linestyle='--')
    ax1.set_xlabel('$(y_{true} - y_{pred}) / y_{true}$')
    
    # Plot overlaid contour plot
    # --------------------------
    plt.figure('Comparison of contours', figsize=(8, 6))
    ax2 = plt.gca()
    ax2.scatter(sobol_samples['x1'], sobol_samples['x2'], color='C1',
                marker='.')
    
    # GP predicted contours
    n_points = 500
    
    # Set up parameter space
    s1_p = np.linspace(bounds['x1'][0], bounds['x1'][1], n_points)
    s2_p = np.linspace(bounds['x2'][0], bounds['x2'][1], n_points)
    x1_p, x2_p = np.meshgrid(s1_p, s2_p)
    
    # GP prediction
    y_gp, y_sigma = gp.predict(np.array([x1_p.flatten(), x2_p.flatten()]).T, 
                               return_std=True)
    
    # Plot
    ax2.contour(x1_p, x2_p, y_gp.reshape(x1_p.shape), contour_levels, colors='C0',
                linestyles='--')
    
    # True contours
    s1_t = np.linspace(bounds['x1'][0], bounds['x1'][1], n_points)
    s2_t = np.linspace(bounds['x2'][0], bounds['x2'][1], n_points)
    x1_t, x2_t = np.meshgrid(s1_t, s2_t)
    
    # True function values
    y_t = d_fcn(x1_t.flatten(), x2_t.flatten())
    
    ax2.contour(x1_t, x2_t, y_t.reshape(x1_t.shape), contour_levels, colors='k', 
                linestyles='-')
    
    # True minima
    if true_minima:
        ax2.plot(x1_min, x2_min, color='r', marker='x', linestyle='None')
    
    # Add legend
    truth_line = mlines.Line2D([], [], color='black',
                               label='True $f(x_1, x_2)$')
    sample_line = mlines.Line2D([], [], color='C1', marker='.', 
                                linestyle='none', label='Train samples')
    mean_line = mlines.Line2D([], [], color='C0', linestyle='--', 
                              label='GP prediction')
    lines = [truth_line, sample_line, mean_line]
    if true_minima:
        minima_line = mlines.Line2D([], [], color='r', marker='x', 
                                    linestyle='none', label='True minima')
        lines.append(minima_line)
    
    legend = ax2.legend(handles=lines, loc='upper right', frameon=True, 
                        bbox_to_anchor=(1.1, 1.1))
    
    # Set labels
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title(f'$n_{{samples}} = ${n_samples}', loc='left')
    
    # Plot the posterior standard deviation
    # -------------------------------------
    plt.figure('Posterior standard deviation', figsize=(8, 6))
    ax3 = plt.gca()
    
    lev = np.linspace(0, 6, 25)
    hc = ax3.contourf(x1_p, x2_p, y_sigma.reshape(x1_p.shape), lev)
    
    for hci in hc.collections:
        hci.set_edgecolor('face')
    
    # Plot samples
    ax3.scatter(sobol_samples['x1'], sobol_samples['x2'], color='C1', 
                marker='.')
    
    # Colorbar
    hcb = plt.colorbar(hc)
    hcb.set_label('Posterior standard deviation')
        
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')    


def branin_function():
    # Plot GP applied to Branin function
    # ----------------------------------
    # Sampled parameter space
    n_samples = 100
    bounds = {'x1': [-5.0, 10.0], 'x2': [0.0, 15.0]}    
    
    # Contour levels
    levels = np.linspace(0, 10, 4)
    levels = np.append(levels, np.linspace(11, 300, 20))
    
    # True minima
    x1_min = [-np.pi, np.pi, 9.42478]
    x2_min = [12.275, 2.275, 2.475]

    apply_gp(df.branin, bounds, n_samples, levels, true_minima=[x1_min, x2_min])
    

def eggbox_function():
    # Plot GP applied to eggbox likelihood
    # ------------------------------------
    # Sampled parameter space
    n_samples = 100
    bounds = {'x1': [0, 10 * np.pi], 'x2': [0.0, 10 * np.pi]}
    
    # Contour levels
    levels = np.linspace(0, 20, 12)
    # levels = np.append(levels, np.linspace(11, 300, 20))

    apply_gp(df.eggbox_loglikelihood, bounds, n_samples, levels)


if __name__ == '__main__':
    plot_utils.set_nes_plot_style()
    
    # Eggbox function toy problem
    eggbox_function()
    
    # Branin function toy problem
    # branin_function()

    

    
