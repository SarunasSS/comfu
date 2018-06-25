# comfu: powerful Python function computing toolkit

## Description
**comfu** is a Python package which provides the tools to create programs using mathematical functions and their operations. The module uses is driven by polynomial functions for which many function operations can be mapped to parameter operations. 
**comfu** can be used: 
- To extend discrete domains algorithms to continuous ones
- Working with unstructured data
- ...

## Usage
First convert all the input functions to the Function objects using the fitting tools in the funcpy.utils module. Once that is done then operate the functions as needed. 

## Features
Currently the following operations on functions are supported for N input N output functions: 
- evaluate 
- +scalar, -scalar, *scalar
- +, -, * 
- integral, derivative
- argument scale ( xmul )
- upscaling to a higher dimension function with a constant dimmension = adddim

## Example
```python
from matplotlib import pyplot as plt
from comfu import Function

n_inputs = 1
n_outputs = 1

f = Function( n_inputs=n_inputs, n_outputs=n_outputs )
g = Function( n_inputs=n_inputs, n_outputs=n_outputs )

x = np.sort( np.random.random( [ 10, n_inputs ] ) - 0.5, 0 )

plt.plot( x.squeeze(), f( x ).squeeze(), label='f' )
plt.plot( x.squeeze(), g( x ).squeeze(), label='g' )
plt.plot( x.squeeze(), ( f + g )( x ).squeeze(), label='f+g' )
plt.plot( x.squeeze(), ( f - g )( x ).squeeze(), label='f-g' )
plt.plot( x.squeeze(), ( f * g )( x ).squeeze(), label='f*g' )
plt.plot( x.squeeze(), f.dx()( x ).squeeze(), label='dfdx' )
plt.plot( x.squeeze(), f.int()( x ).squeeze(), label='integral f' )
plt.plot( x.squeeze(), f.xmul( 0.5, 0 )( x ).squeeze(), label='f( 0.5 * x )' )

plt.legend()

plt.show()
```

## TODO
Core functionalities:   
- translate in argument ie. f( x - b ) = xadd 
- function fitting 
- compute f( g( x ) )
- downscaling to a lower dimension function by slicing on a variable
- bounds to restrict functions to a certain range of arguments

Improvements:
- make approx order different per function
- run on GPU

Make examples:
- PDE
- Q-learning

