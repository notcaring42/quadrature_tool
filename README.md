Pro Quadrature Viewer ZX790
===========================

Pro Quadrature Viewer ZX790 is a graphical utility for visualizing 
[quadrature](http://en.wikipedia.org/wiki/Quadrature_%28mathematics%29 "Quadrature") 
methods on user-inputted mathematical functions.

Requirements
------------
This program is written in and should be invoked with Python 2.7.
It also requires the NumPy, SciPy, Matplotlib, and PyQt4 libraries.

Basic Usage
-----------

To use the program, simply enter a function at the text box labeled __f(x)__. You
can input the function as you would any Python expression, but 
_make sure you use '^' for powers instead of '**'_. Above __f(x)__ a graph will
appear demonstrating quadrature on that function. You can change which quadrature
method will be used in the graph in the box to the right of the graph. X-bounds and
the number of polygons used in quadrature (__n__) can be changed in the box under the 
method-selection box.

Below the input for __f(x)__ is an input for __F(x)__, the antiderivative of __f(x)__.
This input is optionally, but providing an __F(x)__ will also display an error value
for the area calculated by the quadrature method.

On the right-hand side is a table and numerous checkboxes above it. This is for comparing
various quadrature methods to see each method's estimate, the percent error, and the time
each method took to make its estimate. The estimate and time taken will always show up,
but to get the percent error, you'll need to input __F(x)__. Only methods that are checked
above the table will be tested and show up in the table, and you can specify an __n__ value
for each of the methods to use.

Further Info
------------

More information on the program, and quadrature in general, is provided in the User's Manual,
located at __/docs/UserGuide.pdf__.

Credits
-------

All of the coding for the quadrature methods was done by Jorge Almeyda and John Kluesner.
All of the GUI coding was done by William (Max) Mays.
Special thanks to Robert Olsen, the professor of the class for which this project was done.