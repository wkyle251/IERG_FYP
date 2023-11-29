import torch

solvers = {}
# ft, dann, mme
def register_solver(name):
    def decorator(cls):
        solvers[name] = cls
        return cls
    return decorator

def get_solver(name, *args):
    print("solver")
    print(solvers)
    solver = solvers[name](*args)
    return solver