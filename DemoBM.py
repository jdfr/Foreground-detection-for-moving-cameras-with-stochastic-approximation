import numpy as np
import ModelBM as m
import traceback

class ProcessSpec:
  __slots__ = 'VideoFileSpec', 'deltaFrame', 'numFrames', 'SelectedFeatures', 'Epsilon', 'tau', 'MinHessian', 'RansacReproj', 'epsilonRGB', 'coefReplicaBorde'
  def __init__(self):
    pass


#FROM demo.m
def demoConfig(seqId, fetId):
  pspec = ProcessSpec()
  if seqId==1:
    pspec.VideoFileSpec = 'woman/frames/scene%05d.png'
    pspec.deltaFrame = 0
    pspec.numFrames = 557
  elif seqId==2:
    pspec.VideoFileSpec = 'badminton/frames/scene%05d.png'
    pspec.deltaFrame = 0
    pspec.numFrames = 1150
  if fetId==1:
    fets=[3-1, 20-1, 21-1, 22-1]
  elif fetId==2:
    fets=[5-1, 19-1, 20-1, 21-1, 22-1]
  elif fetId==3:
    fets=[19-1, 20-1, 22-1]
  pspec.SelectedFeatures = np.array(fets)
  #Set Epsilon value: 0.002, 0.01, 0.02 or 0.03
  pspec.Epsilon=0.03
  #Set Tau value: 0.999, 0.9995 or 0.9999
  pspec.tau=0.999
  pspec.MinHessian = 400
  pspec.RansacReproj = 3
  pspec.epsilonRGB = 0.2
  pspec.coefReplicaBorde = 0.09
  return pspec


def example1():
  try:
    #Choose the sequence number: 1 or 2
    NdxVideo=1
    
    #Choose the feature set: 1, 2 or 3
    FeatureIdx=3
    
    pspec = demoConfig(NdxVideo, FeatureIdx)
    
    m.simulaSecuenciaParanoramaBM4(pspec.VideoFileSpec,pspec.deltaFrame,pspec.numFrames, pspec.SelectedFeatures, pspec.Epsilon, pspec.MinHessian, pspec.RansacReproj, pspec.epsilonRGB, pspec.coefReplicaBorde,pspec.tau)
  except Exception:
    traceback.print_exc()

example1()
