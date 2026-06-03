# GenerateDynamics
A package for generating some common dynamics on networks, as well as some types of analysis.

Getting started: Packages you will need: numpy, networkx, copy, scipy

Introduction to the methods:
The main classes are the laplacian_dynamics class (to be discussed first), the divergences class, the master_stability class and the laplacian_pseudospectra class. This can be used for diffusive coupling on top of the network. To get started we will first import the package and initialize laplacian_dynamics

```
from GenerateDynamics import laplacian_dynamics
ld = laplacian_dynamics()
```

Now we have a couple of options for getting a graph initialized, one we can explicitely set the graph using set_graph, or the convert_adjacency options
```
ld.set_graph(G)
```
or
https://github.com/jefish003/GenerateDynamics.git``
#A is either a 2d numpy array or a 2d matrix
ld.convert_adjacency(A)
```
These methods will internally store the graph that has been set.
Alternatively, we can set the graph during a call to generate the dynamics, for instance
```
#G is a networkx Graph or DiGraph
x,t = ld.continuous_time_linear_dynamics(G=G)
```
will generate continuous time linear consensus dynamics, using the graph Laplacian (i.e dx/dt = -Lx). NOTE: if a graph has been stored, but the option above is used (G=G), then the internally stored graph will be overwritten by this new graph G.

The options for continuous_time_linear_dynamics are as follows:

G - the graph you want to run dynamics on top of (default None, this is assumed to have already been set by one of the methods above)

tmax - (default tmax = 100) the time to integrate until (starting time is always assumed to be t = 0)

timestep - (default timestep = 0.1) the timestep for integration

init_cond_type - (default 'normal') the distribution to draw the initial condition from. See below for a list of options

init_cond - (default None) if you wish to use your own initial condition you can specify it here, and init_cond_type and init_cond_params will then be ignored

init_cond_params - (default [0,1]) an iterable containing the parameters for the distribution in init_cond_type. (NOTE: MUST BE AN ITERABLE!!!!)

init_cond_offset - (default 0) a constant offset to be added to each value of the initial condition, this is a way to create a shifted distribution if desired



The options for init_cond_type are as follows:

'normal' - the two parameter normal distribution. The first parameter is the mean and the second is the standard deviation

'uniform' - the two parameter uniform distribution. The first parameter is the minimum possible value and the second is the maximum

'laplace' - the two parameter Laplace distribution. The first parameter is the mean, the second parameter is a scale paramter, sometimes referred to as the "diversity"

'exponential' - the one parameter exponential distribution.  The parameter is the mean of the distribution.

'rayleigh' - the one parameter Rayleigh distribution. This is the scale parameter of the distribution

'beta' - the two parameter beta distribution. The two parameters are shape parameters for the distribution.

'gamma' - the two parameter gamma distribution. The first parameter is the shape parameter and the second is the scale parameter

'gumbel' - the two parameter Gumbel distribution. The first parameter is the location parameter, the second is the scale parameter.

'chisquare' - the one parameter Chi^2 distribution. The parameter must be an integer, and is the mean of the distribution

'logistic' - the two parameter logistic distribution. The first parameter is the mean, the second parameter is the scale parameter.

'lognormal' - the two parameter lognormal distribution. The first parameter is the logarithm of the mean, and the second is the logarithm of the standard deviation.

'pareto' - the one parameter pareto distribution. The one parameter is the shape parameter.

'f' - the two parameter f distribution. The first parameter is a pos. integer, the degrees of freedom of the numerator. The second parameter degrees of freedom of the denominator.

'vonmises' - the two parameter Von Mises distribution. The first parameter is the mean (wrapped onto the circle...) and the second parameter is the measure of concentration on the circle.

'wald' - the two parameter Wald distribution (aka inverse Gaussian). The first parameter is the mean, the second parameter is the shape parameter.

'weibull' - the one parameter Weibull distribution. The parameter is the shape parameter.


Beyond just linear dynamics, there is also an option for nonlinear dynamics on networks, in this case assuming identical isolated dynamics so the model for the function continuous_time_nonlinear_dynamics  is $dx_i/dt = f(x_i) + k\sum_i(A_{ij}h(x_i,x_j))$, where A is the adjacency matrix, and h is assumed to be a matrix that couples different components. 

It can simply be called in the following manner, since all arguments have defaults.
```
#G is a networkx Graph or DiGraph
x,t = ld.continuous_time_nonlinear_dynamics(G=G)
```

The options for continuous_time_nonlinear_dynamics are as follows:

G - a networkx Graph or DiGraph (default None)

tmax - (default 100) The maximum time to integrate to.

timestep - (default 0.1) The integration timestep

method - (default 'random') The type of method for generating initial conditions. If it is set to 'random' then the initial conditions will simply be drawn randomly from the distribution specified by init_cond_type. If method = 'normalized' then the initial condition will be normalized, and multiplied by scale (see below)

init_cond_type - (default 'normal') the distribution type to draw the initial condition from

init_cond - (default None) if this is not none, then this initial condition will be used, overriding init_cond_type (see above for options)

init_cond_params - (default [0,1]) an iterable (must be an iterable) containing the parameters for the distribution in init_cond_type

init_cond_offset - (default 0) the offset to shift the distribution of the initial condition

p_norm - (default 2) if method = 'normalized' then this is the norm to normalize the initial condition by. Only p-norms are currently supported (np.inf for the infinity norm)

scale - (default 1) if method = 'normalized' then this is how much to scale the initial condition by.

dynamics_type - (default 'Rossler') the type of dynamics to run on the network. See below for the available options

dynamics_params - (default [0.2,0.2,7]) an iterable containing the parameters for the distribution. For more explanation of the number of parameters needed see below

coupling_matrix - (default None) the coupling matrix which details which components to couple through. The internal default is coupling through the first component. So inputting the correct size coupling matrix can alter this coupling structure.

coupling_strength - (default 1) the coupling_strength (k from the equation above).


Currently available ISOLATED dynamics types:

'Rossler' - the Rossler system $dx/dt = -y-z$, $dy/dt = x+ay$, $dz/dt = b+z(x-c)$. The dynamics_params should be listed as $[a,b,c]$. One suggested example set $[0.2,0.2,7]$

'Lorenz' - The Lorenz system $dx/dt = \sigma(y-z)$, $dy/dt = x(\rho -z) -y$, $dz/dt = xy-\beta z$. The parameters should be listed as $[\sigma,\rho,\beta]$. One suggested example set $[10,28,8/3]$

'Aizawa' - The Aizawa system $dx/dt = (z-b)x-d\cdot y$, $dy/dt = d\cdot x+(z-b)y$, $dz/dt = c+az-z^3/3 -(x^2+y^2)(1+ez)+fz$. The parameters should be listed as $[a,b,c,d,e,f]$. One suggested example set $[0.95,0.7,0.6,3.5,0.25,0.1]$

'Chen-Lee' - The Chen-Lee system (sometimes called the Chen system). $dx/dt = ax-yz$, $dy/dt =by+xz$, $dz/dt = cz+(xy/3)$. The parameters should be listed as $[a,b,c]$. One suggested example $[-5.5,3.5,-1]$.

'VanDerPol' - The VanDer Pol system, $dx/dt = y, dy/dt = \mu(1-x^2)y-x$. The parameters should be listed as $[\mu]$. Suggested example value is $[1.5]$.

'Brusselator' - The Brusselator system, $dx/dt = 1-(a+1)x+bx^2y, dy/dt = ax-bx^2y$. The parameters should be listed as $[a,b]$. Suggested example of values $[2.5,1.12]$.

'Wienbridge' - The Wien bridge system, $dx/dt = -x+y-(ay-by^3+cy^5), dy/dt = -(-x+y-(ay-by^3+cy^5))-y$. The parameters should be listed as $[a,b,c]$. Suggested example of values $[3.234, 2.195, 0.666]$


