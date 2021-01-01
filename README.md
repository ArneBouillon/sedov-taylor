# sedov-taylor

Repository containing all of the code used in the process of writing the report for the assignment on Sedov-Taylor explosions. Each plot in the report can be recreated using the code in this repository.

## `Sedov_Taylor_1D` and `Sedov_Taylor_2D`

[MPI-AMRVAC](http://amrvac.org/) modules used for simulating the explosion using the Euler equations. The 1D case uses spherical symmetry, the 2D case is axisymmetrical. The modules also include Jupyter Notebook files used to plot the results.

## `constructSystem.m`

MATLAB file used for manipulating equations, as elaborated upon in the report.

## `sedov_taylor.py` and `util.py`

Python files used to implement a solver for the self-similar solution to the Sedov-Taylor explosion, as well as for plotting the results. More detailed documentation is found among the code itself.

