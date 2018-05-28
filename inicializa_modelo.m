function model= inicializa_modelo(VideoFileSpec,deltaFrame, SelectedFeatures,EpsionInicial)


% Create the structures of the stochastic approximation model
% disp('Creating the structures of the stochastic approximation model...');
VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+1)));
FeatureFrame=ExtractFeatures(VideoFrame,SelectedFeatures);
model = createBM(FeatureFrame);

model.Z = 50;
model.Epsilon=EpsionInicial;
model.NumPatterns = 2;
% model.aciertosAnt = 1000;

% Allocate scape for the set of images to initialise the model 
FirstFrames = zeros(size(FeatureFrame,1),size(FeatureFrame,2),size(FeatureFrame,3),model.NumPatterns);
FirstFrames(:,:,:,1) = FeatureFrame;

% Store the frames
VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+1)));
FeatureFrame=ExtractFeatures(VideoFrame,SelectedFeatures);
for NdxFrame=2:model.NumPatterns
    FirstFrames(:,:,:,NdxFrame) = FeatureFrame;
end

% disp('Initialising the model...');
% Initialize the model using a set of frames
model = initializeBM_MEX(model,FirstFrames);


% Estimate the noise of the sequence
model.Noise = estimateNoise(model);
model.Mu=squeeze(model.Mu);
model.MuFore=squeeze(model.MuFore);
model.C=squeeze(model.C);

end