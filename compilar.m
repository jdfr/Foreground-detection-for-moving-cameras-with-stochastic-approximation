%You might be tempted to force the loading of the host's OpenCV, as explained here: https://github.com/kyamagu/mexopencv/issues/365 (search for "See the troubleshooting section of the wiki")
%with something like this: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/lib/x86_64-linux-gnu/libgcc_s.so.1 matlab
%If you are able to make it work, I want to know!

%before compiling for linux, please do the following:
%   -locate Matlab's openCV libraries (int Matlab's binary folder) and header files (in the vision toolbox), and make sure that they are the version 3.1.0.
%   -If they aren't, download from the corresponding version of opencv_contrib the surf.* and /*/*.hpp files required for SURF
%   -make symbolic links from all of Matlab's openCV libraries to *.so files, and put them in this folder.
%   -Do the same for the libraries tbb and tbbmalloc. If Matlab at last includes the library tbmalloc_proxy, uncomment the corresponding #includes in the *.cpp files.

%mex -setup C++

ipaths={'-I/PUT/HERE/THE/PATH/TO/EIGEN/INCLUDE/FILES'}; %something like {'-I/home/username/eigen-eigen-f562a193118d'}

ocvst = '-lopencv_';
ocvend = ''; %'248';

calib3d = [ocvst 'calib3d' ocvend];
features2d = [ocvst 'features2d' ocvend];
core = [ocvst 'core' ocvend];
flann = [ocvst 'flann' ocvend];
cudaarithm = [ocvst 'cudaarithm' ocvend];
highgui = [ocvst 'highgui' ocvend];
cudabgsegm = [ocvst 'cudabgsegm' ocvend];
imgcodecs = [ocvst 'imgcodecs' ocvend];
cudacodec = [ocvst 'cudacodec' ocvend];
imgproc = [ocvst 'imgproc' ocvend];
cudafeatures2d = [ocvst 'cudafeatures2d' ocvend];
ml = [ocvst 'ml' ocvend];
cudafilters = [ocvst 'cudafilters' ocvend];
objdetect = [ocvst 'objdetect' ocvend];
cudaimgproc = [ocvst 'cudaimgproc' ocvend];
photo = [ocvst 'photo' ocvend];
cudalegacy = [ocvst 'cudalegacy' ocvend];
shape = [ocvst 'shape' ocvend];
cudaobjdetect = [ocvst 'cudaobjdetect' ocvend];
stitching = [ocvst 'stitching' ocvend];
cudaoptflow = [ocvst 'cudaoptflow' ocvend];
superres = [ocvst 'superres' ocvend];
cudastereo = [ocvst 'cudastereo' ocvend];
videoio = [ocvst 'videoio' ocvend];
cudawarping = [ocvst 'cudawarping' ocvend];
video = [ocvst 'video' ocvend];
cudev = [ocvst 'cudev' ocvend];
videostab = [ocvst 'videostab' ocvend];

matlabpath = '/PUT/HERE/MATLAB/ROOT/PATH'; %something like /usr/local/MATLAB/R2017b

mex('-g', '-compatibleArrayDims', 'updateBM_MEX.cpp', 'BasicMath.cpp', '-L.', '-ltbb', '-ltbbmalloc', ipaths{:})

mex('-g', '-compatibleArrayDims', 'initializeBM_MEX.cpp', 'BasicMath.cpp', ipaths{:})

mex('-g', '-compatibleArrayDims', 'extrae_transformacion_BF2_MEX.cpp', 'surf.cpp', ...
'-D__OPENCV_BUILD', '-L.', ... %the symbol __OPENCV_BUILD is required to build surf.cpp
calib3d, features2d, core, flann, cudaarithm, highgui, cudabgsegm, ...
imgcodecs, cudacodec, imgproc, cudafeatures2d, ml, cudafilters, objdetect, cudaimgproc, photo, ...
cudalegacy, shape, cudaobjdetect, stitching, cudaoptflow, superres, cudastereo, videoio, cudawarping, ...
video, cudev, videostab, ...
ipaths{:}, ['-I' matlabpath '/toolbox/vision/builtins/src/ocvcg/opencv/include'], '-I.')

mex('-g', '-compatibleArrayDims', 'aplica_transformacion6_MEX.cpp', ...
'-L.', '-ltbb', '-ltbbmalloc', ...
calib3d, features2d, core, flann, cudaarithm, highgui, cudabgsegm, ...
imgcodecs, cudacodec, imgproc, cudafeatures2d, ml, cudafilters, objdetect, cudaimgproc, photo, ...
cudalegacy, shape, cudaobjdetect, stitching, cudaoptflow, superres, cudastereo, videoio, cudawarping, ...
video, cudev, videostab, ...
ipaths{:}, ['-I' matlabpath '/toolbox/vision/builtins/src/ocvcg/opencv/include'])
