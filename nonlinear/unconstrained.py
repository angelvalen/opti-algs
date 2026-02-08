import math
import inspect
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import solve_triangular



def uniform_search(f, a, b, eps=1e-6, **kwargs):

    # Initialization
    iters = math.floor(2*(b - a)/eps)
    delta = (b - a) / (iters + 1)
    min_a = a
    min_f = f(a)

    # Process
    for i in range(1, iters + 2):

        a_i = a + i*delta
        f_i = f(a_i)

        if f_i < min_f:
            min_a = a_i
            min_f = f_i
    
    # Output
    return min_a, min_f, iters


def golden_section_search(f, a, b, eps=1e-6, **kwargs):

    # Initialization
    phi = (1 + 5**0.5) / 2
    I = b - a
    xa = b - I/phi
    xb = a + I/phi
    fa = f(xa)
    fb = f(xb)
    iters = math.ceil((math.log(I) - math.log(eps))/math.log(phi))

    # Process
    for i in range(1, iters + 1):

        I = I / phi

        if fa > fb:
            a = xa
            xa = xb
            fa = fb
            xb = a + I/phi
            fb = f(xb)

        else:
            b = xb
            xb = xa
            fb = fa
            xa = b - I/phi
            fa = f(xa)
    
    # Output
    min_x = (a + b)/2
    return min_x, f(min_x), iters


def fibonacci_search(f, a, b, eps=1e-6, **kwargs):
    
    # Initialize
    I = b - a

    # - Generate Fibonacci numbers
    F = [1, 2]
    while F[-1] <= I / eps:
        F.append(F[-1] + F[-2])

    xa = b - I * F[-2] / F[-1]
    xb = a + I * F[-2] / F[-1]
    fa = f(xa)
    fb = f(xb)
    iters = len(F) - 1

    # Process
    for i in range(iters):
        
        I = I * F[iters - i - 1] / F[iters - i]

        if fa > fb:
            a = xa
            xa = xb
            fa = fb
            xb = a + I * F[iters - i - 1] / F[iters - i]
            fb = f(xb)

        else:
            b = xb
            xb = xa
            fb = fa
            xa = b - I * F[iters - i - 1] / F[iters - i]
            fa = f(xa)
    
    # Output
    min_x = (a + b) /2
    return min_x, f(min_x), iters   


def bisection_search(f, df, a, b, eps=1e-6, **kwargs):

    I = b - a

    iters = 0
    while iters < math.log(I/eps, 2):

        x = (b + a)/2
        dfx = df(x)

        if dfx < 0:
            a = x
        elif dfx > 0:
            b = x
        else:
            return x, f(x), iters
        
        iters += 1
    
    min_x = (a + b)/2
    return min_x, f(min_x), iters


def cyclic_coordinate_descent(f, x0, eps=1e-6, **kwargs):
    
    n = len(x0)
    x = x0.copy()
    y = x.copy()
    error = 2*eps
    iters = 0

    while error >= eps:
        for i in range(n):

            ei = np.zeros(n)
            ei[i] = 1
            g = lambda c: f(y + c*ei)
            lambda_ = minimize_scalar(g).x
            y += lambda_*ei
            
        error = np.linalg.norm(x - y)
        x = y.copy()
        iters += 1
    
    return x, f(x), iters


def hooke_jeeves(f, x0, eps=1e-6, **kwargs):

    n = len(x0)
    x = x0.copy()
    y = x.copy()
    error = 2*eps
    iters = 0

    while error >= eps:

        for i in range(n):

            ei = np.zeros(n)
            ei[i] = 1
            g = lambda c: f(y + c*ei)
            lambda_ = minimize_scalar(g).x
            y += lambda_*ei

        if np.linalg.norm(x - y) < eps:
            return x, f(x), iters
        
        else:
            d = y - x
            h = lambda c: f(y + c*d)
            lambda_ = minimize_scalar(h).x
            y += lambda_*d
            error = np.linalg.norm(x - y)
            x = y.copy()
            iters += 1

    return x, f(x), iters


def steepest_descent(f, df, x0, eps=1e-6, **kwargs):

    x = x0.copy()
    g = df(x)
    iters = 0

    while np.linalg.norm(g) >= eps:
        
        h = lambda c: f(x - c*g)
        lambda_ = minimize_scalar(h).x
        x -= lambda_*g
        g = df(x)
        iters += 1
    
    return x, f(x), iters


def newton_method(f, df, ddf, x0, eps=1e-6, **kwargs):

    x = x0.copy()
    g = df(x)
    M = ddf(x)
    iters = 0

    while np.linalg.norm(g) >= eps:
        x -= np.linalg.inv(M).dot(g)
        g = df(x)
        M = ddf(x)
        iters += 1

    return x, f(x), iters


def levenberg_marquardt(f, x0, df, ddf, eps=1e-6, **kwargs):

    n = len(x0)
    lambda_ = 1
    x_prev = x0.copy()
    g = df(x_prev)
    H = ddf(x_prev)
    M = lambda_ * np.identity(n) + H
    L = None
    error = 2*eps
    iters = 0

    while error >= eps:

        while L is None:
            
            try:
                L = np.linalg.cholesky(M)
            
            except np.linalg.LinAlgError:
                lambda_ *= 4

        # Compute new point using Cholesky's matrix
        z = solve_triangular(L, -g, lower=True)
        w = solve_triangular(L.T, z, lower=False)
        x = w + x_prev
        
        diff = x - x_prev
        error = np.linalg.norm(diff)

        # Update lambda
        R = (f(x_prev) - f(x)) / (-g.dot(diff) - 1/2 * (diff).T.dot(H).dot(diff))
        if R < 0.25:
            lambda_ *= 4
        if R > 0.75:
            lambda_ /= 2

        g = df(x)
        H = ddf(x)
        M = lambda_ * np.identity(n) + H
        L = None
        x_prev = x.copy()
        iters += 1

    return x, f(x), iters


def fletcher_reeves_quadratic(c, Q, x0, eps=1e-6, **kwargs):
    
    n = len(x0)
    y = x0.copy()
    f = lambda x: c.T.dot(x) + 1/2 * x.T.dot(Q).dot(x)
    df = c + Q.dot(y)
    d = np.zeros((n, n))
    iters = 0

    while np.linalg.norm(df) >= eps:
        
        # Build conjugate directions & update y
        for j in range(n):
            
            d[j] = - df
            for i in range(j):
                d[j] += df.T.dot(Q).dot(d[i]) / d[i].T.dot(Q).dot(d[i]) * d[i]
            
            lambda_ = - df.T.dot(d[j]) / d[j].T.dot(Q).dot(d[j])
            y += lambda_*d[j]
            df = c + Q.dot(y)
            iters += 1

    return y, f(y), iters


def fletcher_reeves(f, df, x0, eps=1e-6, **kwargs):

    n = len(x0)
    y = x0.copy()
    g = df(y)
    iters = 0

    while np.linalg.norm(g) >= eps:

        d = -g.copy()

        for j in range(n):
            
            h = lambda c: f(y + c*d)
            lambda_ = minimize_scalar(h).x
            y += lambda_*d
            new_g = df(y)

            if j < n - 1:
                alpha = new_g.T.dot(new_g) / g.T.dot(g)
                d = - new_g + alpha*d
            
            g = new_g
            iters += 1
    
    return y, f(y), iters


def polak_ribiere(f, df, x0, eps=1e-6, **kwargs):

    n = len(x0)
    y = x0.copy()
    g = df(y)
    iters = 0

    while np.linalg.norm(g) >= eps:

        d = -g.copy()

        for j in range(n):
            
            h = lambda c: f(y + c*d)
            lambda_ = minimize_scalar(h).x
            y += lambda_*d
            new_g = df(y)

            if j < n - 1:
                alpha = new_g.T.dot(new_g - g) / g.T.dot(g)
                d = - new_g + alpha*d
            
            g = new_g
            iters += 1
    
    return y, f(y), iters


def davidon_fletcher_powell(f, df, x0, D1=None, eps=1e-6, **kwargs):

    n = len(x0)
    y = x0.copy()
    g = df(y)
    D = np.identity(n) if D1 is None else D1.copy()
    iters = 0

    while np.linalg.norm(g) >= eps:

        for j in range(n):

            d = - D.dot(g)
            h = lambda c: f(y + c*d)
            lambda_ = minimize_scalar(h).x
            y += lambda_*d
            new_g = df(y)

            if j < n - 1:
                p = lambda_*d
                q = new_g - g
                D = D + np.outer(p, p) / p.T.dot(q) - D @ np.outer(q, q) @ D / q.T.dot(D).dot(q)
                
            g = new_g
            iters += 1
    
    return y, f(y), iters


def is_valid_test(test, algorithm):

    sig = inspect.signature(algorithm)
    
    required_params = {
        name for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    return required_params.issubset(test.keys())


def run_tests(tests, algorithms):

    for t in tests:

        real_min_point = t["min_point"]
        real_min_value = t["min_value"]
        print("="*10 + f" Function: {t["name"]} " + "="*10)
        print(f"Expected min at x = {real_min_point}, with value f(x) = {real_min_value}\n")
        
        # Running algorithms
        for alg in algorithms:

            if is_valid_test(t, alg):

                print(f"Running {alg.__name__}:")

                point, value, iters = alg(**t)

                print(f"- Approximated optimal point: {point}")
                print(f"- Approximated optimal value: {value}")
                print(f"- Number of iterations: {iters}")

                error = np.linalg.norm(point - real_min_point)
                print(f"Absolute error: {error}\n")


quadratic_1d = {
        "name": "Quadratic 1D: x^2 + 2",
        "f": lambda x: x**2 + 2,
        "df": lambda x: 2*x,
        "min_point": 0,
        "min_value": 2,
        "a": -2,
        "b": 1,
        "eps": 1e-6,
}
quadratic_2d = {
        "name": "Quadratic 2D: (x_1 - 3)^2 + (x_2 - 2)^2",
        "f": lambda x: (x[0]-3)**2 + (x[1]-2)**2,
        "df": lambda x: np.array([2*(x[0] - 3), 2*(x[1] - 2)]),
        "ddf": lambda x: np.array([[2, 0], 
                                   [0, 2]]),
        "min_point": np.array([3, 2]),
        "min_value": 0,
        "x0": np.array([0.0, 0.0]),
        "eps": 1e-6,
}
notes_example_2_2 = {
        "name": "Notes Example 2.2: 3*x_1^2 + 2*x_2^2 + 4*x_1*x_2 - 2*x_1 + 3*x_2 + 1",
        "f": lambda x:  3*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] - 2*x[0] + 3*x[1] + 1,
        "df": lambda x: np.array([6*x[0] + 4*x[1] - 2, 4*x[1] + 4*x[0] + 3]),
        "ddf": lambda x: np.array([[6, 4], 
                                  [4, 4]]),
        "min_point": np.array([2.5, -3.25]),
        "min_value": -6.375,
        "x0": np.array([0.0, 0.0]),
        "eps": 1e-6,
}
rosenbrock_2d = {
        "name": "Rosenbrock 2D: 100*(x_2 - x_1^2)^2 + (1 - x_1)^2",
        "f": lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2,
        "df": lambda x: np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
                                   200*(x[1] - x[0]**2)]),
        "ddf": lambda x: np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], 
                                  [-400*x[0], 200]]),
        "min_point": np.array([1, 1]),
        "min_value": 0,
        "x0": np.array([-1.0, 1.0]),
        "eps": 1e-6,
}
quadratic_fletcher_reeves = {
        "name": "Quadratic for FR: 3*x_1^2 + 2*x_2^2 + 4*x_1*x_2 - 2*x_1 + 3*x_2",
        "c": np.array([-2, 3]),
        "Q": np.array([[6, 4],
                       [4, 4]]),
        "min_point": np.array([2.5, -3.25]),
        "min_value": -7.375,
        "x0": np.array([0.0, 0.0]),
        "eps": 1e-6,
}

TESTS = [quadratic_fletcher_reeves, notes_example_2_2, rosenbrock_2d]
ALGS = [fletcher_reeves_quadratic, fletcher_reeves, davidon_fletcher_powell]

"""
Hacer una test.py donde defina el objeto test, y que pueda elegir que funciones probar y que algortimos testear, 
y que run_tests sea gigante y separe casos de constrained, con intervalos, etc. Para asegurar buen usage.
Y que ponga para cada caso pues intervalos de inicio si hay, o derivadas que se usan si se usan, etc.
"""

if __name__ == "__main__":
    run_tests(TESTS, ALGS)
            