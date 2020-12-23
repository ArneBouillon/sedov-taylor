def inv(f, eta_s):
    return lambda zeta, *args: -f(eta_s - zeta, *args)

def to_function(values, eta_s):
    return lambda eta: values[int(round(eta / eta_s * (len(values) - 1)))]
