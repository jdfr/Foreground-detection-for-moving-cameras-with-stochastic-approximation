# CCode.pyx

cimport numpy as cnp
import numpy as np

cnp.import_array()

from libcpp cimport bool

cimport cython

cdef extern from "BMArgs.h":# namespace "XXX":
  ctypedef struct Args_initializeBM_MEX:
    const char *logFileName
    double *secondArgData
    int secondArgDims[4]
    int *NumCompGauss
    int *CurrentFrame
    int *NumComp
    int *NumCompUnif
    double *Pi
    double *Mu
    double *MuFore
    double *C
    double *Min
    double *Max
    double *Den
  ctypedef struct Args_extrae_transformacion_BF2_MEX:
    double *arg0
    double *arg1
    double *output
    double ransacReproj
    int minHessian
    int tam_cols
    int tam_rows
    int tam_cols_ext
    int tam_rows_ext
    bool upright
  ctypedef struct Args_aplica_transformacion6_MEX:
    double *arg0
    double *arg1
    double *arg2
    double *arg3
    double *arg4
    double *arg5
    double *arg6
    int tam_cols
    int tam_rows
    int tam_objgrid
    int tam_cols_ini
    int tam_rows_ini
    int nchannels
    float *oarg5
    double *oarg4  
    double *oarg2
    double *oarg0
    double *oarg1
    double *oarg3
  ctypedef struct Args_updateBM_MEX:
    bool wantOutput
    const char *logFileName
    int NumImageRows
    int NumImageColumns
    int Dimension 
    double *arg1
    int *NumCompGauss
    int *CurrentFrame
    int *NumComp
    int *NumCompUnif
    int *Z
    double *Epsilon
    double *Pi
    double *Mu
    double *MuFore
    double *C
    double *LogDetC
    double *Min
    double *Max
    double *Den
    double *Counter
    double *Noise
    double *oarg1
    #double *oarg2 #THIS SEEMS UNUSED
  void initializeBM_MEX(Args_initializeBM_MEX *args);
  void extrae_transformacion_BF2_MEX(Args_extrae_transformacion_BF2_MEX *args);
  void aplica_transformacion6_MEX(Args_aplica_transformacion6_MEX *args);
  void updateBM_MEX(Args_updateBM_MEX *args);

#FROM createBM.m
cdef class ModelBM:
  cdef public str Log
  cdef bytes LogBytes
  cdef char* LogCharPtr
  cdef object shapeArg2Update
  #VERY IMPORTANT: THE *.cpp CODE DEPENDS ON NumCompGauss==1!!!!! If it is bigger it will either segfault or do nonsense!!!!! In particular, "Den" must be an array if NumCompGauss>1 !!!!!!
  cdef public double Epsilon, H, Den
  cdef public int NumPatterns, NumCompGauss, NumCompUnif, NumComp, Z, CurrentFrame, KernelProcesses, NumImageRows, NumImageColumns, Dimension
  cdef public cnp.ndarray Min, Max, Noise #cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran']
  cdef public cnp.ndarray Counter #cnp.ndarray[cnp.float64_t, ndim=2, mode='fortran']
  cdef public cnp.ndarray Pi, LogDetC #cnp.ndarray[cnp.float64_t, ndim=3, mode='fortran']
  cdef public cnp.ndarray Mu, MuFore #cnp.ndarray[cnp.float64_t, ndim=4, mode='fortran']
  cdef public cnp.ndarray C, InvC #cnp.ndarray[cnp.float64_t, ndim=5, mode='fortran']
  
  def __init__(self, bool init=True, tuple frameshape=(0,0,0)):
    cdef cnp.npy_intp dims[5]
    if init:
      # R.M.Luque and Ezequiel Lopez-Rubio -- February 2011
      self.NumImageRows = frameshape[0]
      self.NumImageColumns = frameshape[1]
      self.Dimension = frameshape[2]
      self.shapeArg2Update = (self.NumImageRows, self.NumImageColumns, self.NumComp)

      # Epsilon is the step size which regulates how quick the learning process is
      # Valid values are shown in the paper 
      self.Epsilon = 0.01

      self.NumPatterns = 100# Number of used patterns to initilise the model
      self.H = 2 # h is a global smoothing parameter to compute the noise (by default is 2)
      self.NumCompGauss=1# Number of Gaussian distributions (it properly works with 1)
      self.NumCompUnif=1# Number of uniform distributions (it properly works with 1)
      self.Z = 250# Maximum number of consecutive frames in which a pixel belongs to the foreground class 
                    # It is assumed that it is computed offline by analising 
                    # a subset of frames of the sequence (by default 250)
      self.CurrentFrame =1# Indicates the current frame (at the begining 1)
      self.KernelProcesses = 4 # Number of CPU kernels to parallel the process
      #self.Dimension=Dimension # Number of features of each pixel
      
      self.Den = 0

      self.NumComp=self.NumCompGauss+self.NumCompUnif # Total number of distributions
      self.Log = 'temp.txt' # Name of the log file
      self.LogBytes = bytes(self.Log)
      self.LogCharPtr = self.LogBytes

      # Allocating space for work variables
      dims[0] = self.NumComp
      dims[1] = self.NumImageRows
      dims[2] = self.NumImageColumns
      self.Pi = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.Dimension
      self.Min = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 1)
      self.Max = cnp.PyArray_ZEROS(1, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.Dimension
      dims[1] = self.NumCompGauss
      dims[2] = self.NumImageRows
      dims[3] = self.NumImageColumns
      self.Mu = cnp.PyArray_ZEROS(4, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.Dimension
      dims[1] = self.Dimension
      dims[2] = self.NumCompGauss
      dims[3] = self.NumImageRows
      dims[4] = self.NumImageColumns
      self.C = cnp.PyArray_ZEROS(5, dims, cnp.NPY_FLOAT64, 1)
      self.InvC=cnp.PyArray_ZEROS(5, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.NumCompGauss
      dims[1] = self.NumImageRows
      dims[2] = self.NumImageColumns
      self.LogDetC = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.Dimension
      dims[1] = self.NumCompGauss
      dims[2] = self.NumImageRows
      dims[3] = self.NumImageColumns
      self.MuFore = cnp.PyArray_ZEROS(4, dims, cnp.NPY_FLOAT64, 1)
      dims[0] = self.NumImageRows
      dims[1] = self.NumImageColumns
      self.Counter = cnp.PyArray_ZEROS(2, dims, cnp.NPY_FLOAT64, 1)
   
  def clone(self):
    cdef ModelBM model = ModelBM(init=False)
    model.NumImageRows = self.NumImageRows
    model.NumImageColumns = self.NumImageColumns
    model.Dimension = self.Dimension
    model.shapeArg2Update = self.shapeArg2Update

    # Epsilon is the step size which regulates how quick the learning process is
    # Valid values are shown in the paper 
    model.Epsilon = self.Epsilon

    model.NumPatterns = self.NumPatterns# Number of used patterns to initilise the model
    model.H = self.H # h is a global smoothing parameter to compute the noise (by default is 2)
    model.NumCompGauss=self.NumCompGauss# Number of Gaussian distributions (it properly works with 1)
    model.NumCompUnif=self.NumCompUnif# Number of uniform distributions (it properly works with 1)
    model.Z = self.Z# Maximum number of consecutive frames in which a pixel belongs to the foreground class 
                  # It is assumed that it is computed offline by analising 
                  # a subset of frames of the sequence (by default 250)
    model.CurrentFrame =self.CurrentFrame# Indicates the current frame (at the begining 1)
    model.KernelProcesses = self.KernelProcesses # Number of CPU kernels to parallel the process
    #model.Dimension=self.Dimension # Number of features of each pixel

    model.NumComp=self.NumComp# Total number of distributions
    model.Log = self.Log # Name of the log file
    model.LogBytes = self.LogBytes
    model.LogCharPtr = model.LogBytes

    # Allocating space for work variables
    model.Pi=self.Pi.copy(order='F')
    model.Min=self.Min.copy(order='F')
    model.Max=self.Max.copy(order='F')
    model.Den=self.Den
    model.Mu=self.Mu.copy(order='F')
    model.C=self.C.copy(order='F')
    model.InvC=self.InvC.copy(order='F')
    model.LogDetC=self.LogDetC.copy(order='F')
    model.MuFore=self.MuFore.copy(order='F')
    model.Counter=self.Counter.copy(order='F')
    model.Noise=self.Noise.copy(order='F')
    return model
     
  def initializeBM_PYX(self,cnp.ndarray FirstFrames):
    cdef Args_initializeBM_MEX args
    args.logFileName = self.LogCharPtr
    args.NumCompGauss = &(self.NumCompGauss)
    args.CurrentFrame = &(self.CurrentFrame)
    args.NumComp = &(self.NumComp)
    args.NumCompUnif = &(self.NumCompUnif)
    args.Den = &(self.Den)
    if badArray(FirstFrames): raise IndexError('FirstFrames MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Pi): raise IndexError('Pi MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Mu): raise IndexError('Mu MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.MuFore): raise IndexError('MuFore MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.C): raise IndexError('C MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Min): raise IndexError('Min MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Max): raise IndexError('Max MUST BE A FORTRAN ARRAY!!!!')
    cdef tuple shape = (<object>FirstFrames).shape
    cdef int s0 = shape[0]
    cdef int s1 = shape[1]
    cdef int s2 = shape[2]
    cdef int s3 = shape[3]
    args.secondArgDims[0] = s0
    args.secondArgDims[1] = s1
    args.secondArgDims[2] = s2
    args.secondArgDims[3] = s3
    args.secondArgData = <cnp.float64_t *>FirstFrames.data
    args.Pi = <cnp.float64_t *>self.Pi.data
    args.Mu = <cnp.float64_t *>self.Mu.data
    args.MuFore = <cnp.float64_t *>self.MuFore.data
    args.C = <cnp.float64_t *>self.C.data
    args.Min = <cnp.float64_t *>self.Min.data
    args.Max = <cnp.float64_t *>self.Max.data
    initializeBM_MEX(&args)

  def updateBM_PYX(self, cnp.ndarray FirstFrames, bool wantOutput=True):
    cdef cnp.npy_intp dims[3]
    cdef Args_updateBM_MEX args
    cdef cnp.ndarray arg1
    #cdef cnp.ndarray arg2 #THIS SEEMS UNUSED
    args.wantOutput = wantOutput
    args.logFileName = self.LogCharPtr
    args.NumImageRows = self.NumImageRows
    args.NumImageColumns = self.NumImageColumns
    args.Dimension = self.Dimension
    args.NumCompGauss = &(self.NumCompGauss)
    args.CurrentFrame = &(self.CurrentFrame)
    args.NumComp = &(self.NumComp)
    args.NumCompUnif = &(self.NumCompUnif)
    args.Den = &(self.Den)
    args.Z = &(self.Z)
    args.Epsilon = &(self.Epsilon)
    if badArray(FirstFrames): raise IndexError('FirstFrames MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Pi): raise IndexError('Pi MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Mu): raise IndexError('Mu MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.MuFore): raise IndexError('MuFore MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.C): raise IndexError('C MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.LogDetC): raise IndexError('LogDetC MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Min): raise IndexError('Min MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Max): raise IndexError('Max MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Counter): raise IndexError('Counter MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Noise): raise IndexError('Noise MUST BE A FORTRAN ARRAY!!!!')
    args.arg1 = <cnp.float64_t *>FirstFrames.data
    args.Pi = <cnp.float64_t *>self.Pi.data
    args.Mu = <cnp.float64_t *>self.Mu.data
    args.MuFore = <cnp.float64_t *>self.MuFore.data
    args.C = <cnp.float64_t *>self.C.data
    args.LogDetC = <cnp.float64_t *>self.LogDetC.data
    args.Min = <cnp.float64_t *>self.Min.data
    args.Max = <cnp.float64_t *>self.Max.data
    args.Counter = <cnp.float64_t *>self.Counter.data
    args.Noise = <cnp.float64_t *>self.Noise.data
    cdef tuple shape = (<object>FirstFrames).shape
    cdef int s0 = shape[0]
    cdef int s1 = shape[1]
    cdef int s2 = shape[2]
    if wantOutput:
      dims[0] = s0
      dims[1] = s1
      dims[2] = s2
      arg1 = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
      #arg2 = np.zeros(self.shapeArg2Update, dtype=np.float64, order='F') #THIS SEEMS UNUSED
      args.oarg1 = <cnp.float64_t *>arg1.data
      #args.oarg2 = <cnp.float64_t *>arg2.data #THIS SEEMS UNUSED
    updateBM_MEX(&args)
    if wantOutput:
      return arg1#, arg2 #THIS SEEMS UNUSED
  
  def aplica_transformacion6_PYX(self, cnp.ndarray objgridX, cnp.ndarray objgridY, cnp.ndarray R, int tam_cols, int tam_rows, int tam_objgrid, int tam_cols_ini, int tam_rows_ini, int nchannels):
    cdef cnp.npy_intp dims[3]
    cdef Args_aplica_transformacion6_MEX args
    cdef cnp.ndarray TransMu_nD, TransMuFore_nD, TransCounter, TransR_nD, TransPi_nD, Corona
    if badArray(objgridX): raise IndexError('objgridX MUST BE A FORTRAN ARRAY!!!!')
    if badArray(objgridY): raise IndexError('objgridY MUST BE A FORTRAN ARRAY!!!!')
    if badArray(R): raise IndexError('R MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Mu): raise IndexError('Mu MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.MuFore): raise IndexError('MuFore MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Pi): raise IndexError('Pi MUST BE A FORTRAN ARRAY!!!!')
    if badArray(self.Counter): raise IndexError('Counter MUST BE A FORTRAN ARRAY!!!!')
    args.arg0 = <cnp.float64_t *>objgridX.data
    args.arg1 = <cnp.float64_t *>objgridY.data
    args.arg2 = <cnp.float64_t *>self.Mu.data
    args.arg3 = <cnp.float64_t *>self.MuFore.data
    args.arg4 = <cnp.float64_t *>R.data
    args.arg5 = <cnp.float64_t *>self.Pi.data
    args.arg6 = <cnp.float64_t *>self.Counter.data
    args.tam_cols = tam_cols
    args.tam_rows = tam_rows
    args.tam_objgrid = tam_objgrid
    args.tam_cols_ini = tam_cols_ini
    args.tam_rows_ini = tam_rows_ini
    args.nchannels = nchannels
    dims[0] = nchannels
    dims[1] = tam_objgrid
    dims[2] = 1
    TransMu_nD = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
    TransMuFore_nD = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
    dims[0] = tam_objgrid
    dims[1] = 1
    TransCounter = cnp.PyArray_ZEROS(2, dims, cnp.NPY_FLOAT64, 1)
    dims[0] = nchannels*nchannels
    dims[1] = tam_objgrid
    dims[2] = 1
    TransR_nD = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
    dims[0] = 2
    dims[1] = tam_objgrid
    dims[2] = 1
    TransPi_nD = cnp.PyArray_ZEROS(3, dims, cnp.NPY_FLOAT64, 1)
    dims[0] = tam_objgrid
    dims[1] = 1
    Corona = cnp.PyArray_ZEROS(2, dims, cnp.NPY_FLOAT32, 1)
    args.oarg0 = <cnp.float64_t *>TransMu_nD.data
    args.oarg1 = <cnp.float64_t *>TransMuFore_nD.data
    args.oarg2 = <cnp.float64_t *>TransCounter.data
    args.oarg3 = <cnp.float64_t *>TransR_nD.data
    args.oarg4 = <cnp.float64_t *>TransPi_nD.data
    args.oarg5 = <cnp.float32_t *>Corona.data
    aplica_transformacion6_MEX(&args)
    return TransMu_nD, TransMuFore_nD, TransCounter, TransR_nD, TransPi_nD, Corona


def extrae_transformacion_BF2_PYX(cnp.ndarray arg0, cnp.ndarray arg1, bool upright, int minHessian, double ransacReproj, int tam_cols, int tam_rows, int tam_cols_ext, int tam_rows_ext):
  cdef cnp.npy_intp dims[2]
  cdef Args_extrae_transformacion_BF2_MEX args
  cdef cnp.ndarray output
  if badArray(arg0): raise IndexError('arg0 MUST BE A FORTRAN ARRAY!!!!')
  if badArray(arg1): raise IndexError('arg1 MUST BE A FORTRAN ARRAY!!!!')
  args.arg0 = <cnp.float64_t *>arg0.data
  args.arg1 = <cnp.float64_t *>arg1.data
  dims[0] = 3
  dims[1] = 3
  output = cnp.PyArray_ZEROS(2, dims, cnp.NPY_FLOAT64, 1)
  args.ransacReproj = ransacReproj
  args.minHessian = minHessian
  args.upright = upright
  args.tam_cols = tam_cols
  args.tam_rows = tam_rows
  args.tam_cols_ext = tam_cols_ext
  args.tam_rows_ext = tam_rows_ext
  args.output = <cnp.float64_t *>output.data
  extrae_transformacion_BF2_MEX(&args)
  return output
  
cdef bool badArray(cnp.ndarray a):
  #the constant should be NPY_ARRAY_FARRAY, but cython used the deprecated NPY_FARRAY
  return (cnp.PyArray_FLAGS(a) & cnp.NPY_FARRAY) != cnp.NPY_FARRAY
