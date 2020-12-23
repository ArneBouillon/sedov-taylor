import scipy as sp
import scipy.integrate as integ
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

import util

num_calls = 0


def eq(eta, y, gamma):
    A, B, C = y

    A_ = (A*(16*A*C**3 - 10*A*C**2*gamma - 10*A*C**2 + A*C*gamma**2 + 2*A*C*gamma + A*C - 6*B*gamma**2 + 6*B))/(eta*(gamma - 2*C + 1)*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma))
    B_ = -(4*B**2*gamma - 4*B**2*gamma**2 + 8*A*B*C**2*gamma + 8*A*B*C**2 - A*B*C*gamma**2 - 15*A*B*C*gamma - 14*A*B*C + 5*A*B*gamma**2 + 10*A*B*gamma + 5*A*B)/(eta*(A + 2*A*gamma + 2*B*gamma + 4*A*C**2 + A*gamma**2 - 2*B*gamma**2 - 4*A*C - 4*A*C*gamma))
    C_ = -(8*A*C**3 - 14*A*C**2 - 6*B + 6*B*gamma**2 + 5*A*C + 10*A*C*gamma + 12*B*C*gamma + 5*A*C*gamma**2 - 14*A*C**2*gamma - 12*B*C*gamma**2)/(2*(A*eta - 4*A*C*eta + 2*A*eta*gamma + 2*B*eta*gamma + 4*A*C**2*eta + A*eta*gamma**2 - 2*B*eta*gamma**2 - 4*A*C*eta*gamma))
    
    return np.array([A_, B_, C_])

def jac(eta, y, gamma):
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
            ]
        ]
    )

def energy_integrand(A, B, C, eta_s):
    return lambda eta: (B(eta) + A(eta) * C(eta)**2) * eta**4

def energy(A, B, C, gamma, eta_s):
    const = 32 * sp.pi / (25 * (gamma**2 - 1))
    return const * integ.quad(energy_integrand(A, B, C, eta_s), .001, eta_s, limit=500)[0]

def sedov_taylor(gamma, eta_s, plot=False, plot_integrand=False):
    global num_calls
    num_calls += 1
    
    offset = .001
    eta_s_ = eta_s - offset

    t = np.linspace(offset, eta_s_, 1000)
    y0 = [1., 1., 1.]
    
    sol_inv = integ.odeint(util.inv(eq, eta_s), y0, t, args=(gamma,), tfirst=True, Dfun=util.inv(jac, eta_s))
    sol = sol_inv[::-1,:]
    
    A_ = util.to_function(sol[:,0].copy(), eta_s)
    A = lambda eta: A_(eta - offset)
    B_ = util.to_function(sol[:,1].copy(), eta_s)
    B = lambda eta: B_(eta - offset)
    C_ = util.to_function(sol[:,2].copy(), eta_s)
    C = lambda eta: C_(eta - offset)

    eta = offset;
    n = len(sol[:,1]) - 1
    for i in range(n+1):
        sol[i,1] *= (eta / eta_s_)**2
        sol[i,2] *= (eta / eta_s_)
        
        eta += eta_s / n;


    if plot:
        # plt.plot(t, sol[:,0], '-')
        # plt.plot(t, sol[:,1], '--')
        # plt.plot(t, sol[:,2], '-.')
        # plt.xlabel(r'$\eta$')
        # plt.ylabel(r'Value relative to $\eta = \eta_s$')
        # plt.legend([r'$\rho$', '$p$', '$v$'])
        # plt.savefig('etafs.png', bbox_inches='tight')
        
        Adata = sol[:,0]
        Adata *= (gamma + 1) / (gamma - 1)
        
        plt.plot(list(t * .55**.4)+[list(t * .55**.4)[-1], 1], list(Adata) + [1, 1])
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\rho$')
        plt.savefig('recrho.png', bbox_inches='tight')
    elif plot_integrand:
        l = np.linspace(0, eta_s, 2000)
        plt.plot(l, [*map(energy_integrand(A, B, C, eta_s), l)])
        plt.xlabel(r'$\eta$')
        plt.ylabel('Integrand')
        plt.savefig('integrand.png', bbox_inches='tight')

    return energy(A, B, C, gamma, eta_s)

def find_eta_s(gamma, tol=.0001, eta_s0=1.):
    eta_s = 1
    e = sedov_taylor(gamma, eta_s)
    if e > 1:
        upper = eta_s
        while e > 1:
            eta_s /= 2
            e = sedov_taylor(gamma, eta_s)
        lower = eta_s
    else:
        lower = eta_s
        while e < 1:
            eta_s *= 2
            e = sedov_taylor(gamma, eta_s)
        upper = eta_s
        
    while upper - lower > tol:
        eta_s = (upper + lower) / 2

        e = sedov_taylor(gamma, eta_s)
        if e > 1:
            upper = eta_s
        else:
            lower = eta_s

    return eta_s

def find_eta_s_opt(gamma, tol=.0001, eta_s0=1.):
    return opt.root_scalar(lambda eta_s: sedov_taylor(gamma, eta_s) - 1, x0=eta_s0, x1=eta_s0+.1, rtol=tol).root # TODO: Robust
    


if __name__ == '__main__':
    # gamma = 5/3
    # eta_s = find_eta_s_opt(gamma)
    # sedov_taylor(gamma, eta_s, plot=True)
    
    eta_s_s = [*map(lambda gamma: print(gamma) or find_eta_s_opt(gamma), np.linspace(1.2, 2, 100))]
    plt.plot(np.linspace(1.2, 2, 100), eta_s_s)
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$\eta_s$')
    plt.savefig('etass.png', bbox_inches='tight')
