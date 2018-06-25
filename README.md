# funcpy: powerful Python function computing toolkit

## Description
**funcpy** is a Python package which provides the tools to create programs using mathematical functions and their operations. The module uses is driven by polynomial functions for which many function operations can be mapped to parameter operations. 
Funcpy can be used: 
- To extend discrete domains algorithms to continuous ones
- Working with unstructured data

## Usage
First convert all the input functions to the Function objects using the fitting tools in the funcpy.utils module. Once that is done then operate the functions as needed. 

## Features
Currently the following operations on functions are supported for N input N output functions: 
    -evaluate 
    -+scalar, -scalar, *scalar
    -+, -, * 
    -integral, derivative
    -argument scale ( xmul )
    -upscaling to a higher dimension function with a constant dimmension = adddim

## Examples
	tbd

## TODO
Core functionalities:   
    N input, N output - translate, scale in argument ie. f( a x - b ) = xadd 
    function fitting 
    compute f( g( x ) )
    downscaling to a lower dimension function by slicing on a variable
	bounds to restrict functions to a certain range of arguments

Improvements:
    make approx order different per function
    run on GPU

Make examples:
	PDE
	Q-learning

