# Foreground detection for moving cameras with stochastic approximation

This is a modification of the code for the paper entitled **Foreground detection for moving cameras with stochastic approximation**,
adapted to run in Matlab R2017b for Linux.

The MathWorks provides Matlab binaries as isolated from the host as possible, compiled
in a custom environment, incompatible with the host toolchain. It also ships with a
constellation of third-party libraries built for that environment, so if you want to
make MEX files to use libraries compiled in the host system that are also present
inside Matlab, a universe of pain awaits you for all but the simplest use cases.
Do not go down that road...

So, Matlab R2017b with the vision toolbox ships with OpenCV-3.1. This code only uses SURF
from OpenCV-contrib's nonfree module, and that algorithm is pretty well isolated from
nonfree's other stuff. So, it was a matter of adapting the C++ code to use OpenCV-3.X APIs
and carefully extracting the SURF code and its supporting header files (which are to be
redistributed under their own license, of course!).

The code also uses Intel's TBB, using all of its three libraries (tbb, tbbmalloc and tbbmalloc_proxy).
Unfortunately, Matlab ships with its own version of tbb and tbbmalloc, but not tbbmalloc_proxy.
The easiest option, and the one choosed here, is to disable the use of the latter library by commenting
out all instances of `#include <tbb/tbbmalloc_proxy.hpp>`.

Pretty sure this code can work also for any version of Matlab for Windows/Linux shipping
with OpenCV-3.1, with minimal adaptations.

Example data not included, but can be found at
[Ezequiel's research webpage](http://www.lcc.uma.es/~ezeqlr/nonpan/nonpan.html).

Dependencies:
* Matlab R2017b for Linux
* Eigen, the C++ matrix header-only library

