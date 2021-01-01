"""
Some utility functions that aren't specific to Sedov-Taylor explosions
"""

def inv(f, max_val):
    """
    Return a function that is equivalent to `f`, but subtracts its
     first argument from `max_val` before passing the result as the
     first argument, and negates the result of the call to `f`.
    """
    return lambda zeta, *args: -f(max_val - zeta, *args)

def to_function(values, max_val, min_val=0):
    """
    Convert the given list of `values` into a function that indexes
     into this list based on its argument's value relative to the
     given `max_val` and (optionally) `min_val`.
     
    This is comparable to making a discrete series continuous using
     a zero order hold (ZOH) technique.
    """
    return lambda eta: values[int(round((eta - min_val) / (max_val - min_val) * (len(values) - 1)))]
