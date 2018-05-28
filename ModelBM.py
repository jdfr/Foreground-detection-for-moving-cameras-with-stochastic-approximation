import numpy as np

import UtilBM as u
import CCodeBM as cc

import matplotlib.pyplot as plt

import math

#FROM inicializa_modelo.m
def inicializa_modelo(VideoFileSpec,deltaFrame, SelectedFeatures,EpsionInicial):

  # Create the structures of the stochastic approximation model
  # disp('Creating the structures of the stochastic approximation model...')
  imname = VideoFileSpec % (deltaFrame+1)
  VideoFrame = u.readImg(imname)
  FeatureFrame=u.ExtractFeatures(VideoFrame,SelectedFeatures)

  model = cc.ModelBM(frameshape=FeatureFrame.shape)

  model.Z = 50
  model.Epsilon=EpsionInicial
  model.NumPatterns = 2
  # model.aciertosAnt = 1000

  # Allocate scape for the set of images to initialise the model 
  FirstFrames = np.zeros((FeatureFrame.shape[0],FeatureFrame.shape[1],FeatureFrame.shape[2],model.NumPatterns), order='F')
  FirstFrames[:,:,:,:] = FeatureFrame.reshape((FeatureFrame.shape[0],FeatureFrame.shape[1],FeatureFrame.shape[2],1))

  # disp('Initialising the model...')
  # Initialize the model using a set of frames
  model.initializeBM_PYX(FirstFrames)

  # Estimate the noise of the sequence
  model.Noise = u.estimateNoise(model)
  model.Mu=np.squeeze(model.Mu)
  model.MuFore=np.squeeze(model.MuFore)
  model.C=np.squeeze(model.C)
  
  return model


def simulaSecuenciaParanoramaBM4(VideoFileSpec,deltaFrame,numFrames, SelectedFeatures, Epsilon, MinHessian, RansacReproj, epsilonRGB, coefReplicaBorde,tau):

  modelRGB=inicializa_modelo(VideoFileSpec,deltaFrame, [0,1,2],epsilonRGB)
  modelFeature=inicializa_modelo(VideoFileSpec,deltaFrame, SelectedFeatures,Epsilon)

  numFallos=0
  
  plt.figure()

  for NdxFrame in range(2,numFrames+1):
      imname = VideoFileSpec % (deltaFrame+NdxFrame)
      VideoFrame = u.readImg(imname)
      VideoFrame_0_1=VideoFrame/255
      
      transformacion, fallo=panorama_extrae_transformacion8(VideoFrame,modelRGB, MinHessian, RansacReproj, coefReplicaBorde)
      
      if not fallo:
          FeatureFrame=u.ExtractFeatures(VideoFrame,SelectedFeatures)

          modelRGB_Ant=modelRGB.clone()
        
          fallo=panorama_aplica_transformacion13(VideoFrame_0_1,modelRGB,transformacion,tau)

          if not fallo:
              fallo=panorama_aplica_transformacion13(FeatureFrame,modelFeature,transformacion,tau)
              if not fallo:
                  imMask = modelFeature.updateBM_PYX(FeatureFrame, True)
                  modelRGB.updateBM_PYX(VideoFrame_0_1, False)
                
              else: #Si hay fallo en la segunda aplicacion, deshacemos cambios en RGB y volvemos a estado del frame anterior
                  modelRGB=modelRGB_Ant
          #Si hay fallo aplicando transformacion en RGB no se realizan cambios, luego no hay que deshacer 
      
      if fallo:
          imMask=np.ones(VideoFrame.shape, order='F')
          numFallos=numFallos+1
          
          #Si llevamos 3 frames sin acertar, reiniciamos los modelos con el frame actual
          if numFallos >= 3:
              modelRGB=inicializa_modelo(VideoFileSpec, deltaFrame+NdxFrame-1, [0,1,2],epsilonRGB)
              modelFeature=inicializa_modelo(VideoFileSpec, deltaFrame+NdxFrame-1, SelectedFeatures,Epsilon)
              print 'Severe Error: Reset models'
              numFallos=0
      else:
          numFallos=0
            
      plt.subplot(1,2,1);
      plt.imshow(VideoFrame_0_1);
      plt.title('Original frame %d' % NdxFrame)
      plt.subplot(1,2,2);
      plt.imshow(1 - imMask);
      plt.title('Output');
      plt.show(block=False)
      plt.pause(0.001);
      print "processed frame %d" % NdxFrame

class Transformacion:
  __slots__ = 'xmin', 'xmax', 'ymin', 'ymax', 'tamInicial', 'gridsize', 'objgridX', 'objgridY'
  def __init__(self):
    pass

def panorama_extrae_transformacion8(imActual,model, minHessian, ransacReproj, coefReplicaBorde):

    #Preallocate cell arrays
    I      = [None,None]
    
    valido = False
    fallo = False
    cont = 0
    while not(valido) and (cont < 10):
        #Controla le numero de ejecuciones del bucle, si supera un limite
        #sale devolviendo fallo
        cont=cont+1
        
        #Read file
        #we make a copy so the array we pass to C++ is contiguous
        I[0] = u.shiftdim(np.array(imActual, dtype=np.uint8, order='F'),2,1).copy(order='F')     #0 a 255 uint8
        #Calculate SURF descriptor

        #Read file
        I[1] = np.array(model.Mu*255, dtype=np.uint8, order='F')
        #tamInicial = np.ceil(np.array(I[1].shape)*coefReplicaBorde)
        tamInicial1 = int(math.ceil(I[1].shape[1]*coefReplicaBorde))
        tamInicial2 = int(math.ceil(I[1].shape[2]*coefReplicaBorde))
        numPads = np.zeros((I[1].ndim,), dtype=np.int64)
        numPads[1] = tamInicial1 #tamInicial[1]
        numPads[2] = tamInicial2 #tamInicial[2]
        I[1] = u.padarray_replicate_both(I[1], numPads)
        #Calculate SURF descriptor 
        H = cc.extrae_transformacion_BF2_PYX(I[0], I[1], True, minHessian, ransacReproj, I[0].shape[1], I[0].shape[2], I[1].shape[1], I[1].shape[2]) #Usando BruteForce Matcher, mas rapido
        
        #Obtenemos los extremos de la imagen transformada para saber como
        #de grande tendra que ser BigImage
                
        gridxmax = I[1].shape[2]
        gridymax = I[1].shape[1]

        b = np.array([[1, 1, 1], [gridxmax, 1, 1] , [gridymax, 1, 1], [gridxmax, gridymax, 1]], order='F').T
        transformedCorners = np.linalg.solve(H, b)
        transformedCorners[0,:]=transformedCorners[0,:]/transformedCorners[2,:]
        transformedCorners[1,:]=transformedCorners[1,:]/transformedCorners[2,:]
        
        #Update the maximum and minimum coordinates
        xmax=transformedCorners[0,:].max()
        ymax=transformedCorners[1,:].max()
        xmin=transformedCorners[0,:].min()
        ymin=transformedCorners[1,:].min()
    
        xmin = int(np.ceil(xmin))
        xmax = int(np.floor(xmax))
        ymin = int(np.ceil(ymin))
        ymax = int(np.floor(ymax))
      
        #Controlamos que no haya habido una transformacion erronea

        X=(xmax-xmin)  
        Y=(ymax-ymin)
        y=imActual.shape[0]
        x=imActual.shape[1]

        crecimientoErroneo =((0.6*x>X) or (2*x<X) or (0.6*y>Y) or(2*y<Y))            

        valido = (not(crecimientoErroneo)) and ((-ymin+2)>1) and ((-xmin+2)>1)
        if not(valido):
            print 'Minor error' ##Fallo leve, en extraccion
    
    if not(valido):
        fallo = True
        transformacion = None
    else:
            
        transformacion = Transformacion()
        transformacion.xmin=xmin
        transformacion.xmax=xmax
        transformacion.ymin=ymin
        transformacion.ymax=ymax
        transformacion.tamInicial = np.array([tamInicial1, tamInicial2])
        
        biggridX,biggridY = np.mgrid[xmin:xmax+1,ymin:ymax+1]
        #biggridX = biggridX.T
        #biggridY = biggridY.T
        
        transformacion.gridsize = (biggridX.shape[1], biggridX.shape[0])
        #transformedgrid = np.matmul(np.column_stack((biggridX.flatten(order='F'), biggridY.flatten(order='F'), np.ones((biggridX.size,)))), H.T).astype(np.float32, order='F')
        #if flatten in 'C' order, we can omit the .T's above!!!
        transformedgrid = np.matmul(np.column_stack((biggridX.flatten(), biggridY.flatten(), np.ones((biggridX.size,)))), H.T).astype(np.float32, order='F')
        transformacion.objgridX = transformedgrid[:,0] / transformedgrid[:,2]
        transformacion.objgridY = transformedgrid[:,1] / transformedgrid[:,2]
    return transformacion, fallo


def panorama_aplica_transformacion13(imActual,model, transformacion, tau):

    ##Inicializacion de vbles que se calcularon en extrae_informacion
    xmin=transformacion.xmin
    xmax=transformacion.xmax
    ymin=transformacion.ymin
    ymax=transformacion.ymax
    nchannels = imActual.shape[2]
    tamInicial=transformacion.tamInicial   #Numero de pixeles extendidos [Alto x Ancho]
    objgridX=transformacion.objgridX
    objgridY=transformacion.objgridY
    gridsize=transformacion.gridsize
    ######################################  
    ##Calculamos el R a partir de C y de Mu
    R=(model.C + u.multiprod1001(model.Mu,model.Mu)).reshape((nchannels*nchannels, model.Mu.shape[1], model.Mu.shape[2]), order='F')
    
    fallo=False    
    
    TamBigImages =  (ymax-ymin+1, xmax-xmin+1) #dimensiones del marco del frame actual
    PosImActual_y = slice(-ymin, -ymin+imActual.shape[0]) #arreglado para numpy (empezando a contar por 0)
    PosImActual_x = slice(-xmin, -xmin+imActual.shape[1]) #arreglado para numpy (empezando a contar por 0)
  
    TransMu_nD, TransMuFore_nD, TransCounter, TransR_nD, TransPi_nD, Corona = model.aplica_transformacion6_PYX(objgridX,objgridY, R, tamInicial[1], tamInicial[0], objgridX.shape[0], model.Mu.shape[1], model.Mu.shape[2], nchannels)

    TransMu_nD=TransMu_nD.flatten(order='F').reshape((nchannels, gridsize[0],gridsize[1]), order='F')
    TransMuFore_nD=TransMuFore_nD.flatten(order='F').reshape((nchannels, gridsize[0],gridsize[1]), order='F')
    TransCounter=TransCounter.flatten(order='F').reshape((gridsize[0],gridsize[1]), order='F')
    TransPi_nD=TransPi_nD.flatten(order='F').reshape((2, gridsize[0],gridsize[1]), order='F')
    Corona=Corona.flatten(order='F').reshape((gridsize[0],gridsize[1]), order='F')
    TransR_nD=TransR_nD.reshape((nchannels, nchannels, gridsize[0],gridsize[1]), order='F')
    
    ##Todos los Big... almacenan los resultados finales
    ##los inicializamos a los valores del frame anterior transformado
    
    #[1] raw assignment like these have different semantics in numpy and matlab: in numpy they copy a reference to the same object, in matlab it means a different object with CoW. Fortunately, for this algorithm both semantics have the same results!

    BigImageMuFore = TransMuFore_nD #see [1]
    BigImageCounter = TransCounter #see [1]
    BigImagePi = TransPi_nD #see [1]
    
    BigImageMu = np.zeros((nchannels,)+TamBigImages, order='F')
    #En BigImageMu colocamos el frame actual:
    BigImageMu[:,PosImActual_y,PosImActual_x] = u.shiftdim(imActual,2,1)
    
    Pixeles_En_Frame_Act=np.zeros(TamBigImages, dtype=np.bool_, order='F')
    Pixeles_En_Frame_Act[PosImActual_y,PosImActual_x]= True
    
    #Comprobamos que la tranformacion se ha realizado bien en .mex
    if (TamBigImages[0]!=Corona.shape[0]) or (TamBigImages[1]!=Corona.shape[1]):    #Se trata de evitar errores debidos a malos emparejamientos=> no se actualiza ese frame
        'Fallo en actualizacion (3): se ignora frame'     
        fallo = True
        return fallo
    
    ################################################################
    ###Actualizamos los pixeles correspondientes al borde extendido:  

    ##Calculamos las responsabilities
    DifferencesMatrix=BigImageMu-TransMu_nD

    #This block of code has been moved to avoid an unnecessary CoW (triggered when modifying BigImageMu right after doing aux=BigImageMu)
    #En BigImageMu machacamos la parte de Mu anterior que cae en el frame actual:
    idxFrmAnt=np.logical_and(Pixeles_En_Frame_Act, Corona<=0.5)
    BigImageMu[:,idxFrmAnt]= TransMu_nD[:,idxFrmAnt]
    ##Calculamos el C transformado a partir de R y de Mu transformados
    C_aux=TransR_nD-u.multiprod1001(TransMu_nD,TransMu_nD)
    BigImageC=C_aux #see [1]
    
    DistMahalMatrix=np.zeros((C_aux.shape[2], C_aux.shape[3]), order='F')
    LogDetCMatrix=np.zeros((C_aux.shape[2], C_aux.shape[3]), order='F')
    
    Pixeles_En_Borde_matrix=np.logical_and(Pixeles_En_Frame_Act, Corona>0.5)
    tupleindices_orig= np.nonzero(Pixeles_En_Borde_matrix)
    indices = np.ravel_multi_index(tupleindices_orig, Pixeles_En_Borde_matrix.shape, order='F')
    tupleindices_processed = np.unravel_index(indices, DistMahalMatrix.shape, order='F')
    diagonal=np.diag(model.Noise.flatten(1))  
    
    C_aux_reshaped = C_aux.reshape((C_aux.shape[0], C_aux.shape[1], -1), order='F')
    DifferencesMatrix_reshaped = DifferencesMatrix.reshape((DifferencesMatrix.shape[0], -1), order='F')
    
    for ind in range(len(indices)):
        idx = indices[ind]      
        Sigma = C_aux_reshaped[:,:,idx]+diagonal
        dffsubm = DifferencesMatrix_reshaped[:,idx]
        i0 = tupleindices_processed[0][ind]
        i1 = tupleindices_processed[1][ind]
        #as dffsubm is a 1-D vector, in numpy there is no need to transpose it when it is the first operand of the multiplication, as opposed to the matlab code
        DistMahalMatrix[i0,i1] = np.matmul(dffsubm, np.linalg.solve(Sigma, dffsubm))
        LogDetCMatrix[i0,i1] = np.log(np.linalg.det(Sigma))
    MyLogDensityMatrix=-0.918938533204673*model.Dimension-0.5*LogDetCMatrix-0.5*DistMahalMatrix
    #in this case, for numpy code, there is no need of shiftdim (works the same as squeeze, only that numpy indexing already removes singleton dimensions in this case)
    APriori=TransPi_nD[0,:,:]
    Numerator=APriori*np.exp(MyLogDensityMatrix)
    ResponsibilitiesMatrix=Numerator/(Numerator+(1-APriori))
    
    ##Zona nueva => Calculamos si el color es similar al del borde anterior
    ##extendido

    idxCopiarModeloBorde=np.logical_and(ResponsibilitiesMatrix>tau, Pixeles_En_Borde_matrix)
    BigImageMu[:, idxCopiarModeloBorde]=TransMu_nD[:, idxCopiarModeloBorde]
    
    ##Zona nueva => Si el color es diferente al del borde anterior
    ##extendido, ponemos valores por defecto
    idxResetModeloBorde=np.logical_and(ResponsibilitiesMatrix<=tau, Pixeles_En_Borde_matrix)
    BigImageCounter[idxResetModeloBorde]=0
    BigImagePi[:,idxResetModeloBorde]= 0.5
    
    BigImageC[:,:,idxResetModeloBorde]=0
    for ch in range(nchannels):
        BigImageC[ch,ch,idxResetModeloBorde]=model.Noise[ch]
    BigImageMuFore[:,idxResetModeloBorde]=0.5

    #######################################
    ##Copiamos resultados finales al modelo
    model.Mu[:,:,:] = BigImageMu[:, PosImActual_y,PosImActual_x]
    model.MuFore[:,:,:] = BigImageMuFore[:, PosImActual_y,PosImActual_x]
    model.C[:,:,:,:] = BigImageC[:,:, PosImActual_y,PosImActual_x]
    model.Pi[:,:,:] = BigImagePi[:,PosImActual_y,PosImActual_x]
    model.Counter[:,:] = BigImageCounter[ PosImActual_y,PosImActual_x]
