function simulaSecuenciaParanoramaBM4(VideoFileSpec,deltaFrame,numFrames, ...
    SelectedFeatures, Epsilon, MinHessian, RansacReproj, epsilonRGB, coefReplicaBorde,tau)

modelRGB=inicializa_modelo(VideoFileSpec,deltaFrame, [1 2 3],epsilonRGB);
modelFeature=inicializa_modelo(VideoFileSpec,deltaFrame, SelectedFeatures,Epsilon);

numFallos=0;

for NdxFrame=2:numFrames
    VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+NdxFrame)));
    VideoFrame_0_1=VideoFrame/255;
    
    [transformacion fallo]=panorama_extrae_transformacion8(VideoFrame,modelRGB, MinHessian, RansacReproj, coefReplicaBorde);
    if fallo==false
        FeatureFrame=ExtractFeatures(VideoFrame,SelectedFeatures);

        modelRGB_Ant=modelRGB;
       
        [modelRGB fallo]=panorama_aplica_transformacion13(VideoFrame_0_1,modelRGB,transformacion,tau);

        if fallo==false
            [modelFeature fallo]=panorama_aplica_transformacion13(FeatureFrame,modelFeature,transformacion,tau);
            if fallo==false
                [modelFeature,imMask]=updateBM_MEX(modelFeature,FeatureFrame);
                modelRGB=updateBM_MEX(modelRGB,VideoFrame_0_1);
               
            else %Si hay fallo en la segunda aplicacion, deshacemos cambios en RGB y volvemos a estado del frame anterior
                modelRGB=modelRGB_Ant;
            end
        end  %Si hay fallo aplicando transformacion en RGB no se realizan cambios, luego no hay que deshacer 
    end
    
    if fallo==true
        imMask=ones(size(VideoFrame));
        numFallos=numFallos+1;
        
        %%Si llevamos 3 frames sin acertar, reiniciamos los modelos con el frame actual
        if numFallos >= 3
            modelRGB=inicializa_modelo(VideoFileSpec, deltaFrame+NdxFrame-1, [1 2 3],epsilonRGB);
            modelFeature=inicializa_modelo(VideoFileSpec, deltaFrame+NdxFrame-1, SelectedFeatures,Epsilon);
            'Severe Error: Reset models'
            numFallos=0;
        end
    else
        numFallos=0;
    end   
           
    
    subplot(1,2,1);
    imshow(VideoFrame/255);
    title(sprintf('Original frame %d',NdxFrame));
    subplot(1,2,2);
    imshow(1 - imMask);
    title('Output');
    pause(0.001);
end

end




