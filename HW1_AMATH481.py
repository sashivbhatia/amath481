import numpy as np # type: ignore

#Part 1
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

#Newton Raphson calculator function
def newton_raphson(x0, tol=1e-6, max_iter=100):
    x_vals = [x0]
    for i in range(max_iter):
        x_new = x_vals[-1] - (f(x_vals[-1]) / df(x_vals[-1]))
        x_vals.append(x_new)
        if abs(f(x_new)) < tol:
            break
    return x_vals, len(x_vals) 

#Bisection calculator function
def bisection(a, b, tol=1e-6, max_iter=100):
    mid_vals = []
    for i in range(max_iter):
        mid = (a + b) / 2
        mid_vals.append(mid)
        if abs(f(mid)) < tol:
            break
        elif f(mid) * f(a) < 0:
            b = mid
        else:
            a = mid
    return mid_vals, len(mid_vals)

#Lots of answers for part 1
A1, newton_iterations = newton_raphson(-1.6)
A2, bisection_iterations = bisection(-0.7, -0.4)
A3 = [newton_iterations, bisection_iterations]

#Part 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

#Lots of part 2 answers
A4 = A + B
A5 = 3 * x - 4 * y
A6 = A @ x
A7 = B @ (x - y)
A8 = D @ x
A9 = (D @ y) + z
A10 = A @ B
A11 = B @ C
A12 = C @ D

print(A1)
print(A2)
print(A3)
print(A4)
print(A5)
print(A6)
print(A7)
print(A8)
print(A9)
print(A10)
print(A11)
print(A12)