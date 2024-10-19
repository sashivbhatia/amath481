import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

#variables
L = 4
initial_slope = 0.1
eigenval_guesses = [1, 3, 5, 7, 9]
num_req = 5

def harmonic_oscillator(x, y, ep):
    return [y[1], (x**2 - ep) * y[0]]

def boundary_condition_magic_calculator(ep, initial_slope, L):
    y0 = [0, initial_slope]
    sol = solve_ivp(harmonic_oscillator, [-L, L], y0, args=(ep,), 
                    t_eval=[L], method='RK45', rtol=1e-5, atol=1e-5)
    return sol.y[0][-1]

def adjust_bracket(initial_slope, L, lg, ug):
    fl = boundary_condition_magic_calculator(lg, initial_slope, L)
    fu = boundary_condition_magic_calculator(ug, initial_slope, L)

    while fl * fu > 0:
        ug += 1
        fu = boundary_condition_magic_calculator(ug, initial_slope, L)

    return lg, ug

def find_eigenvalue(initial_slope, L, eigenval_guess):
    lg = 0
    ug = eigenval_guess
    lg, ug = adjust_bracket(initial_slope, L, lg, ug)
    
    sol = root_scalar(boundary_condition_magic_calculator, args=(initial_slope, L),
                      bracket=[lg, ug], method='bisect', rtol=1e-5)
    return sol.root

def compute_eigenfunction(ep, initial_slope, L):
    y0 = [0, initial_slope]
    x_span = np.linspace(-L, L, num=int((L - (-L)) / 0.1) + 1)
    sol = solve_ivp(harmonic_oscillator, [-L, L], y0, args=(ep,),
                    t_eval=x_span, method='RK45', rtol=1e-5, atol=1e-5)
    return sol.t, sol.y[0]

def normalize_eigenfunction(x, eigenfunction):
    norm = np.trapz(eigenfunction**2, x)
    return eigenfunction / np.sqrt(norm)

#solutions
A1 = np.zeros((len(np.linspace(-L, L, num=int((L - (-L)) / 0.1) + 1)), num_req))
A2 = np.zeros(num_req)

for n in range(num_req):
    eigenvalue = find_eigenvalue(initial_slope, L, eigenval_guesses[n])
    A2[n] = eigenvalue
    x_vals, eigenfunction = compute_eigenfunction(eigenvalue, initial_slope, L)
    eigenfunction_normalized = normalize_eigenfunction(x_vals, np.abs(eigenfunction))
    A1[:, n] = eigenfunction_normalized

A1= np.abs(A1)

print("eigenfunctions")
print(A1)
print("eigenvals")
print(A2)

plt.figure(figsize=(10, 6))
for n in range(num_req):
    plt.plot(x_vals, A1[:, n], label=f'Eigenfunction {n+1}')
plt.xlabel('x')
plt.ylabel(r'$\phi_n(x)$')
plt.title('First 5 Normalized Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()