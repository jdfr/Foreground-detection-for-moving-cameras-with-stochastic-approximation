#python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sources = [
  "CCodeBM.pyx",
  "BasicMath.cpp",
  "initializeBM_MEX.cpp",
  "extrae_transformacion_BF2_MEX.cpp",
  "aplica_transformacion6_MEX.cpp",
  "updateBM_MEX.cpp"
  ]

includes = [
  ".",
  "../opencv-2.4.8/include",
  "../opencv-2.4.8/modules/core/include",
  "../opencv-2.4.8/modules/nonfree/include",
  "../opencv-2.4.8/modules/highgui/include",
  "../opencv-2.4.8/modules/imgproc/include",
  "../opencv-2.4.8/modules/video/include",
  "../opencv-2.4.8/modules/features2d/include",
  "../opencv-2.4.8/modules/flann/include",
  "../opencv-2.4.8/modules/calib3d/include",
  "../opencv-2.4.8/modules/objdetect/include",
  "../opencv-2.4.8/modules/legacy/include",
 ]

libs = [
  "opencv_calib3d",
  "opencv_core",
  "opencv_features2d",
  "opencv_flann",
  #"opencv_gpu",
  "opencv_imgproc",
  "opencv_highgui",
  "opencv_ml",
  "opencv_nonfree",
  "opencv_objdetect",
  #"opencv_ocl",
  #"opencv_photo",
  #"opencv_stitching",
  #"opencv_superres",
  "opencv_video",
  #"opencv_videostab",
  "tbb",
  "tbbmalloc",
  #"tbbmalloc_proxy",
  ]

libdirs = ["../opencv-2.4.8/build/lib"]
rtlibdirs = libdirs

setup(ext_modules=[Extension("CCodeBM", sources, 
                             language="c++",
                             include_dirs=includes,
                             libraries=libs,
                             library_dirs=libdirs,
                             runtime_library_dirs=rtlibdirs)],
      cmdclass = {'build_ext': build_ext})
