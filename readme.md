# Foreground detection for moving cameras with stochastic approximation

This is a port of the code for the paper entitled **Foreground detection for moving cameras with stochastic approximation**,
adapted to work in Python 2.7 (some late version, such as 2.7.15).

It is easy to port the C++ code to use newer versions of OpenCV. However, most Debian-based distros do not come with official, 
off-the-shelf packages for OpenCV-contrib's nonfree module, so in most cases you'd need either of:
* Download a binary package from an untrusted source
* Compile OpenCV and OpenCV-contrib from scratch
* Compile just the nonfree module, or just the SURF sources.

And the C++ sources might have to be modified depending on what version of OpenCV is available!
For now, this version works just fine if you compile OpenCV-2.4.8 in the parent folder (`..`).

The original code also uses Intel's TBB, using all of its three libraries (tbb, tbbmalloc and tbbmalloc_proxy).
Unfortunately, there seem to be some incompatibilities that prevent the use of tbb_malloc_proxy. Might be
solvable, but does not warrant the effort right now. The easiest option, and the one chosen here, is to
disable the use of the latter library by commenting out all instances of `#include <tbb/tbbmalloc_proxy.hpp>`.

Example data not included, but can be found at [Ezequiel's research webpage](http://www.lcc.uma.es/~ezeqlr/nonpan/nonpan.html).

Dependencies:
* cython 0.28.2
* numpy 1.14.3
* scipy 1.1.0
* imageio 2.3.0
* matplotlib 2.2.2
* OpenCV 2.4.8
* Intel's TBB
* Eigen, the C++ matrix header-only library

