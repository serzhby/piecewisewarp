piecewisewarp
=============

Cython implementation of piecewise affine warp

How to build
============

Download the code, cd to a project root and run:

	python setup.py build_ext --inplace
	
This command will create file `pwa.so` that may be imported 
into Python directly.
	
Example
=======

You can find example in example/example.py. Try it out with:

    cd example
	python example.py

Result of the warp will be stored in `result.bmp`.
