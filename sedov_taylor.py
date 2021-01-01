"""
The main file that contains the functions for obtaining a self-similar
 solution to the Sedov-Taylor problem. When run as `'__main__'`, their
 usage is illustrated through an example for `gamma` equal to `5/3`.

There are a variable and some undocumented keyword parameters that allow
 a user to recreate the plots from my report. However, these would be
 removed in a production-type release of the code. As such, the code
 making these plots is not always very high-quality; it only serves
 to produce good plots.
"""

import scipy as sp
import scipy.integrate as integ
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22}) # The base font size is not very legible

import util

def eq(eta, y, gamma):
    """
    Return the derivatives of A, B and C (`y`) for the given values of
     `eta` and `gamma`.
    """
    A, B, C = y

    A_ = (A*(16*A*C**3 - 10*A*C**2*gamma - 10*A*C**2 + A*C*gamma**2 + 2*A*C*gamma + A*C - 6*B*gamma**2 + 6*B))/(eta*(gamma - 2*C + 1)*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma))
    B_ = -(4*B**2*gamma - 4*B**2*gamma**2 + 8*A*B*C**2*gamma + 8*A*B*C**2 - A*B*C*gamma**2 - 15*A*B*C*gamma - 14*A*B*C + 5*A*B*gamma**2 + 10*A*B*gamma + 5*A*B)/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma))
    C_ = -(8*A*C**3 - 14*A*C**2 - 6*B + 6*B*gamma**2 + 5*A*C + 10*A*C*gamma + 12*B*C*gamma + 5*A*C*gamma**2 - 14*A*C**2*gamma - 12*B*C*gamma**2)/(2*(A*eta - 4*A*C*eta + 2*A*eta*gamma + 2*B*eta*gamma + 4*A*C**2*eta + A*eta*gamma**2 - 2*B*eta*gamma**2 - 4*A*C*eta*gamma))
    
    return np.array([A_, B_, C_])

def jac(eta, y, gamma):
    """
    Return the Jacobian of the derivatives of A, B and C (`y`) for the
     given values of `eta` and `gamma`.
    """
    A, B, C = y
    
    return np.array(
        [
            [
                (64*A**2*C**5 - 104*A**2*C**4*gamma - 104*A**2*C**4 + 60*A**2*C**3*gamma**2 + 120*A**2*C**3*gamma + 60*A**2*C**3 - 14*A**2*C**2*gamma**3 - 42*A**2*C**2*gamma**2 - 42*A**2*C**2*gamma - 14*A**2*C**2 + A**2*C*gamma**4 + 4*A**2*C*gamma**3 + 6*A**2*C*gamma**2 + 4*A**2*C*gamma + A**2*C - 64*A*B*C**3*gamma**2 + 64*A*B*C**3*gamma + 40*A*B*C**2*gamma**3 - 40*A*B*C**2*gamma - 4*A*B*C*gamma**4 - 4*A*B*C*gamma**3 + 4*A*B*C*gamma**2 + 4*A*B*C*gamma + 12*B**2*gamma**4 - 12*B**2*gamma**3 - 12*B**2*gamma**2 + 12*B**2*gamma)/(eta*(gamma - 2*C + 1)*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                -(2*A**2*(gamma - 1)*(8*C**2*gamma - C*gamma**2 - 7*C*gamma - 6*C + 3*gamma**2 + 6*gamma + 3))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                (4*A**2*(16*A*C**3 - 10*A*C**2*gamma - 10*A*C**2 + A*C*gamma**2 + 2*A*C*gamma + A*C - 6*B*gamma**2 + 6*B))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2) + (A**2*(48*C**2 - 20*C*gamma - 20*C + gamma**2 + 2*gamma + 1))/(eta*(gamma - 2*C + 1)*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)) + (2*A*(16*A*C**3 - 10*A*C**2*gamma - 10*A*C**2 + A*C*gamma**2 + 2*A*C*gamma + A*C - 6*B*gamma**2 + 6*B))/(eta*(gamma - 2*C + 1)**2*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma))
            ],
            [
                (2*B**2*gamma*(gamma - 1)*(8*C**2*gamma - C*gamma**2 - 7*C*gamma - 6*C + 3*gamma**2 + 6*gamma + 3))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                -(5*A + 10*A*gamma + 8*B*gamma + 8*A*C**2 + 5*A*gamma**2 - 8*B*gamma**2 - 14*A*C - 15*A*C*gamma - A*C*gamma**2 + 8*A*C**2*gamma)/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)) - (2*B*gamma*(gamma - 1)*(5*A + 10*A*gamma + 4*B*gamma + 8*A*C**2 + 5*A*gamma**2 - 4*B*gamma**2 - 14*A*C - 15*A*C*gamma - A*C*gamma**2 + 8*A*C**2*gamma))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                -(A*B*(6*A + 17*A*gamma - 12*B*gamma + 24*A*C**2 + 15*A*gamma**2 + 3*A*gamma**3 - A*gamma**4 - 2*B*gamma**2 + 12*B*gamma**3 + 2*B*gamma**4 - 24*A*C - 28*A*C**2*gamma**2 - 32*A*C*gamma + 8*A*C*gamma**2 - 4*A*C**2*gamma + 16*A*C*gamma**3 + 32*B*C*gamma**2 - 32*B*C*gamma**3))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2)
            ],
            [
                (B*(gamma - 1)*(10*C**2*gamma**2 - 16*C**3*gamma + 22*C**2*gamma + 12*C**2 - C*gamma**3 - 14*C*gamma**2 - 25*C*gamma - 12*C + 3*gamma**3 + 9*gamma**2 + 9*gamma + 3))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                -(A*(gamma - 1)*(10*C**2*gamma**2 - 16*C**3*gamma + 22*C**2*gamma + 12*C**2 - C*gamma**3 - 14*C*gamma**2 - 25*C*gamma - 12*C + 3*gamma**3 + 9*gamma**2 + 9*gamma + 3))/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2),
                -(32*A**2*C**4 - 64*A**2*C**3*gamma - 64*A**2*C**3 + 60*A**2*C**2*gamma**2 + 120*A**2*C**2*gamma + 60*A**2*C**2 - 28*A**2*C*gamma**3 - 84*A**2*C*gamma**2 - 84*A**2*C*gamma - 28*A**2*C + 5*A**2*gamma**4 + 20*A**2*gamma**3 + 30*A**2*gamma**2 + 20*A**2*gamma + 5*A**2 + 56*A*B*C*gamma**3 - 48*A*B*C*gamma**2 - 56*A*B*C*gamma + 48*A*B*C - 22*A*B*gamma**4 + 2*A*B*gamma**3 + 46*A*B*gamma**2 - 2*A*B*gamma - 24*A*B + 24*B**2*gamma**4 - 48*B**2*gamma**3 + 24*B**2*gamma**2)/(2*eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma)**2)
            ],
        ]
    )

def energy_integrand(A, B, C):
    """
    Return a function that yields the value of the integrand in
     the modified Sedov-Taylor energy equation using the given
     functions `A`, `B` and `C` for an argument `eta`.
    """
    return lambda eta: (B(eta) + A(eta) * C(eta)**2) * eta**4

def energy_integrand_points(A, B, C, etas):
    """
    Return an array of discretisations of the integrand in the
     modified Sedov-Taylor energy equation, based on the discrete
     values of `A`, `B` and `C` at the values `eta`.
    """
    return B * A * C**2 * etas**4

def energy(gamma, eta_s, offset, sol, etas, method='simps'):
    """
    Return the left-hand side of the modified Sedov-Taylor energy
     equation using the given functions `A`, `B` and `C` and the
     constants `gamma` and `eta_s`, starting the integration from
     `offset` instead of from `0`.
     
    For the correct value of `eta_s` for the given `gamma`, this
     should return `1`.
     
    The `method` parameter dictates how the integration is carried
     out. If it is equal to `'quad'`, a ZOH-like approximation of
     A, B and C is made and integrated using `scipy.integrate.quad`.
     If it is equal to `'simps'`, the composite Simpson's rule is
     used.
    """
    const = 32 * sp.pi / (25 * (gamma**2 - 1))
    
    if method == 'quad':
        A = util.to_function(sol[:,0], eta_s, offset)
        B = util.to_function(sol[:,1], eta_s, offset)
        C = util.to_function(sol[:,2], eta_s, offset)
        
        return const * integ.quad(energy_integrand(A, B, C), offset, eta_s, limit=500)[0]

    elif method == 'simps':
        return const * integ.simps(energy_integrand_points(sol[:,0], sol[:,1], sol[:,2], etas), etas)
    
    else:
        raise ValueError(f'`method` should be "quad" or "simps", but was "{method}" instead.')

def sedov_taylor(gamma, eta_s, offset=.001, points=1000, energy_args={}, plot=False, plot_integrand=False):
    """
    Calculate a self-similar solution to the Sedov-Taylor problem
     using the given values for `gamma` and `eta_s`. The domain
     considered runs from `offset` to `eta_s`, and `points` points
     within it are used.
     
    The return value is a tuple consisting of the left-hand side
     of the modified energy equation (see the `energy` function)
     as well as the calculated `A`, `B` and `C` functions.
     
    When calling `energy`, `energy_args` are passed on as keyword
     parameters.
    """
    range_size = eta_s - offset

    # `t` is the range [`0`, `range_size`], because it is `eta_s` - [`offset`, `eta_s`]
    t = np.linspace(0, range_size, points)
    y0 = [1., 1., 1.]
    
    # The `odeint` call doesn't actually use `Dfun` in this case, since
    #  it does not consider the system stiff. However, it is kept for
    #  reference and to make switching to another integrator easier.
    sol_inv = integ.odeint(util.inv(eq, eta_s), y0, t, args=(gamma,), tfirst=True, Dfun=util.inv(jac, eta_s))
    sol = sol_inv[::-1,:]

    e = energy(gamma, eta_s, offset, sol, np.linspace(offset, eta_s, points), **energy_args)

    A = util.to_function(sol[:,0], eta_s, offset)
    B = util.to_function(sol[:,1], eta_s, offset)
    C = util.to_function(sol[:,2], eta_s, offset)
    if plot_integrand:
        l = np.linspace(0, eta_s, 2000)
        plt.plot(l, [*map(energy_integrand(A, B, C, eta_s), l)])
        plt.xlabel(r'$\eta$')
        plt.ylabel('Integrand')
        plt.savefig('integrand.png', bbox_inches='tight')
        plt.clf()

    if plot:
        sol = sol.copy()
        multiplier = np.linspace(offset, eta_s, points)
        sol[:,1] *= multiplier**2
        sol[:,2] *= multiplier
        
        plt.plot(t, sol[:,0], '-')
        plt.plot(t, sol[:,1], '--')
        plt.plot(t, sol[:,2], '-.')
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'Value relative to $\eta = \eta_s$')
        plt.legend([r'$\rho$', '$p$', '$v$'])
        plt.savefig('etafs.png', bbox_inches='tight')
        plt.clf()

        Adata = sol[:,0]
        Adata *= (gamma + 1) / (gamma - 1)

        plt.plot(list(t * .55**.4)+[list(t * .55**.4)[-1], 1], list(Adata) + [1, 1])
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\rho$')
        plt.savefig('recrho.png', bbox_inches='tight')
        plt.clf()

    return e, A, B, C

def find_eta_s_old(gamma, tol=.0001, eta_s0=1.):
    """
    Return the correct value for eta_s for the given value of
     `gamma` using a tolerance of `tol` and a starting guess
     of `eta_s0`.
     
    This function uses a quick and naive bisection algorithm
     I implemented to get some first results. Usage of this
     function is not recommended; `find_eta_s` is much more
     efficient.
    """
    eta_s = 1
    e = sedov_taylor(gamma, eta_s)[0]
    if e > 1:
        upper = eta_s
        while e > 1:
            eta_s /= 2
            e = sedov_taylor(gamma, eta_s)[0]
        lower = eta_s
    else:
        lower = eta_s
        while e < 1:
            eta_s *= 2
            e = sedov_taylor(gamma, eta_s)[0]
        upper = eta_s
        
    while upper - lower > tol:
        eta_s = (upper + lower) / 2

        e = sedov_taylor(gamma, eta_s)[0]
        if e > 1:
            upper = eta_s
        else:
            lower = eta_s

    return eta_s

def find_eta_s(gamma, tol=.0001, eta_s0=1., eta_s1=1.1, st_args={}, verbose=False):
    """
    Return the correct value for eta_s for the given value of
     `gamma` using a tolerance of `tol` and starting guesses
     of `eta_s0` and `eta_s1`. When calling `sedov_taylor`,
     `st_args` are passed on as parameters.

    If `verbose` is truthy, more information about the root
     finding result is printed to STDOUT.
    """

    result = opt.root_scalar(lambda eta_s: sedov_taylor(gamma, eta_s, **st_args)[0] - 1, x0=eta_s0, x1=eta_s1, rtol=tol)

    if verbose:
        print("find_eta_s:")
        print(f"  {result.function_calls} function calls made")
        print(f"  {result.iterations} iterations done")
        print(f"  Converged? {result.converged}")
        print(f"  Exit flag: {result.flag}")

    elif not result.converged:
        print("find_eta_s:")
        print("  Warning: `scipy.optimize.root_scalar` has not converged!")

    return result.root


if __name__ == '__main__':
    PLOT = False # Set this to `True` to recreate some of the plots from the report

    gamma = 5/3
    eta_s = find_eta_s(gamma)
    sedov_taylor(gamma, eta_s, plot=PLOT, plot_integrand=PLOT)

    if PLOT:
        eta_s_s = [*map(lambda gamma: print(gamma) or find_eta_s(gamma), np.linspace(1.2, 2, 100))]
        plt.plot(np.linspace(1.2, 2, 100), eta_s_s)
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\eta_s$')
        plt.savefig('etass.png', bbox_inches='tight')
        plt.clf()

        for method in ('quad', 'simps'):
            pointss = [*map(lambda p:int(p) if int(p) % 2 else int(p) + 1, 10**np.linspace(1, 4, 100))]
            eta_s_s = [*map(lambda points: print(points) or find_eta_s(5/3, st_args={'points': points, 'energy_args': {'method': method}}), pointss)]
            diffs = [abs(x - eta_s_s[-1]) for x in eta_s_s[:-1]]
            plt.plot(pointss[:-1], diffs, '-' if method == 'quad' else '--')

        plt.xlabel(r'Points')
        plt.ylabel(r'$|\eta_{s, points} - \eta_s|$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(["'quad'", "'simps'"])
        plt.savefig('pointss.png', bbox_inches='tight')
        plt.clf()
