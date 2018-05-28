import numpy as np
import scipy.signal as ssig
import imageio as i

def readImg(imname):
  return np.array(i.imread(imname), dtype=np.float64, order='F')

#FROM ExtractFeatures.m
def ExtractFeatures(VideoFrame,SelectedFeatures):
  NumRowsImg=VideoFrame.shape[0]
  NumColsImg=VideoFrame.shape[1]
  NumFeatures=len(SelectedFeatures)
  FeatureFrame=np.zeros((NumRowsImg,NumColsImg,NumFeatures), order='F')
  for NdxFeature in range(NumFeatures):
      MyFeature=SelectedFeatures[NdxFeature]
      if MyFeature>=0 and MyFeature<=2:
              # Red, green and blue channels
              FeatureFrame[:,:,NdxFeature]=VideoFrame[:,:,MyFeature]/255
      elif MyFeature>=3 and MyFeature<=5:
              # Normalized red, green and blue channels
              SumFrame=VideoFrame.sum(axis=2)
              SumFrame[SumFrame==0]=1
              FeatureFrame[:,:,NdxFeature]=VideoFrame[:,:,MyFeature-3]/SumFrame
      elif MyFeature>=6 and MyFeature<=11:
              # Haar-like features considered in Han & Davis (2012)
              SumFrame=VideoFrame.sum(axis=2)
              MyFilter=HFilter(MyFeature-6)
              #equivalences SCIPY <=> MATLAB: 
              #              signal.convolve2d(img, np.rot90(fil), mode='same') <=> imfilter(img, rot90(fil, 2), 0, 'conv')
              #              signal.correlate2d(img, fil, mode='same') <=> imfilter(img, fil, 0, 'corr')
              FeatureFrame[:,:,NdxFeature]=ssig.correlate2d(SumFrame, MyFilter, mode='same')/(3*255*np.abs(MyFilter).sum())
      elif MyFeature==12:
              # Gradient in the horizontal direction, considered in Han & Davis (2012)
              SobelFilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]], order='F')
              SumFrame=VideoFrame.sum(axis=2)
              FeatureFrame[:,:,NdxFeature]=ssig.correlate2d(SumFrame, SobelFilter, mode='same')/(3*255*np.abs(SobelFilter).sum())
      elif MyFeature==13:
              # Gradient in the vertical direction, considered in Han & Davis (2012)
              SobelFilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]], order='F').T
              SumFrame=VideoFrame.sum(axis=2)
              FeatureFrame[:,:,NdxFeature]=ssig.correlate2d(SumFrame, SobelFilter, mode='same')/(3*255*np.abs(SobelFilter).sum())
      elif MyFeature==14:
              # Red channel of the current pixel, normalized with the current pixel 
              # and the pixel immediately to the left of the current one
              #ShiftedFrame=circshift(VideoFrame,[1 0 0])
              ShiftedFrame=np.roll(VideoFrame,1, axis=0)
              SumFrame=np.sum(VideoFrame+ShiftedFrame,axis=2)
              SumFrame[SumFrame==0]=1
              FeatureFrame[:,:,NdxFeature]=6*VideoFrame[:,:,1]/SumFrame     
      elif MyFeature==15:
              # Green channel of the pixel immediately to the lower right of the
              # current one, normalized with the current pixel
              ShiftedFrame=np.roll(VideoFrame,(-1,-1), axis=(0,1))
              SumFrame=VideoFrame.sum(axis=2)
              SumFrame[SumFrame==0]=1
              FeatureFrame[:,:,NdxFeature]=3*ShiftedFrame[:,:,1]/SumFrame 
      elif MyFeature>=16 and MyFeature<=18:
              # Red, green and blue channels, median filtered
              FeatureFrame[:,:,NdxFeature]=ssig.medfilt(VideoFrame[:,:,MyFeature-16], (5,5))/255
      elif MyFeature>=19 and MyFeature<=21:
              # Normalized red, green and blue channels, median filtered
              SumFrame=VideoFrame.sum(axis=2)
              SumFrame[SumFrame==0]=1
              NormFrame=VideoFrame[:,:,MyFeature-19]/SumFrame
              FeatureFrame[:,:,NdxFeature]=ssig.medfilt2d(32768*NormFrame,(5,5))/32768
      elif MyFeature==22:
              # Tiny Haar-like feature (I)
              MyFilter=np.array([[1,0,1],[1,0,1],[1,0,1]], order='F')
              SumFrame=VideoFrame.sum(axis=2)
              FeatureFrame[:,:,NdxFeature]=ssig.correlate2d(SumFrame,MyFilter,mode='same')/(3*6*255)             
      elif MyFeature==23:
              # Tiny Haar-like feature (II)
              MyFilter=np.array([[1,1,1],[1,0,1],[1,1,1]], order='F')
              SumFrame=VideoFrame.sum(axis=2)
              FeatureFrame[:,:,NdxFeature]=ssig.correlate2d(SumFrame,MyFilter)/(3*8*255)                             
  return FeatureFrame



#FROM ExtractFeatures.m
# Filters corresponding to the Haar-like features considered in:
# Han, B. and Davis, L.S. (2012). Density-Based Multifeature Background
# Subtraction with Support Vector Machine. IEEE Transactions on Pattern
# Analysis and Machine Intelligence 34(5), 1017-1023.
def HFilter(NdxFilter):
  if NdxFilter==0 or NdxFilter==1:
    fil = [
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1]]
  elif NdxFilter==2:
    fil = [
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1],
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1],
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1],
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1]]
  elif NdxFilter==3 or NdxFilter==4:
    fil = [
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0],
    [1, 1, 1, 1, 0.5, 0, 0, 0, 0]]
  elif NdxFilter==5:
    fil = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]]
  return np.array(fil, dtype=np.float64, order='F')

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m = (shape[0]-1.)/2.
    n = (shape[1]-1.)/2.
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

SmoothingFilter = matlab_style_gauss2D((3,3),0.5)

def estimateNoise(model):
  # noise = estimateNoise(model)
  # This function compute the noise of a sequence from the structure 
  # previously computed 'model'.  

  # R.M.Luque and Ezequiel Lopez-Rubio -- February 2011

  # The mean of the scene is used as the original frame
  MuImage = np.array(np.squeeze(shiftdim(model.Mu,2,1)), dtype=np.float64, order='F')

  # The smoothing approach is applied
  SmoothFrame=np.zeros(MuImage.shape, order='F')
  for idx in range(MuImage.shape[2]):
    SmoothFrame[:,:,idx] = ssig.correlate2d(MuImage[:,:,idx],SmoothingFilter, mode='same')

  # The difference between the two images is obtained
  dif = np.square(MuImage - SmoothFrame)

  # A 0.01-winsorized mean is applied instead of the standard mean because
  # the first measure is more robust and certain extreme values are removed
  dif2 = dif.reshape((dif.shape[0]*dif.shape[1],model.Dimension), order='F')
  dif3 = np.sort(dif2,axis=0)
  idx = int(np.round(np.max(dif3.shape)*0.99))
  for NdxDim in range(model.Dimension):
      dif3[idx:,NdxDim] = dif3[idx-2,NdxDim]

  noise = np.mean(dif3,axis=0)
  return noise

def shiftdim(x, n=None, nargout=2):
    outsel = slice(nargout) if nargout > 1 else 0
    x = np.asanyarray(x)
    s = x.shape
    m = next((i for i, v in enumerate(s) if v > 1), 0)
    if n is None:
        n = m
    if n > 0:
        n = n % x.ndim
    if n > 0:
        if n <= m:
            x = x.reshape(s[n:])
        else:
            x = x.transpose(np.roll(range(x.ndim), -n))
    elif n < 0:
            x = x.reshape((1,)*(-n)+x.shape)
    return (x, n)[outsel]

def padarray_replicate_both(x, numPads):
  numPads = np.column_stack((numPads, numPads))
  padded = np.pad(x, numPads, 'edge')
  return padded.copy(order='F')

#c=multiprod(a,b,[1 0] [0 1]) with size(a)==size(b) seems to be equivalent to:
#   size(a)==[a1 a2...]
#   aa=reshape(a, a1, 1, a2, ...)
#   bb = reshape(b, 1, a1, ...)
#   c =     bsxfun(@times, a, b) =>this step is implicit in numpy expansion of singleton dimensions
def multiprod1001(a,b):
  sa = a.shape
  sb = b.shape
  sa = (sa[0], 1)+sa[1:]
  sb = (1,)+sb
  return a.reshape(sa)*b.reshape(sb)