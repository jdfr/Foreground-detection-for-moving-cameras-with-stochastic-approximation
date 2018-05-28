function noise = estimateNoise(model)
% noise = estimateNoise(model)
% This function compute the noise of a sequence from the structure 
% previously computed 'model'.  

% R.M.Luque and Ezequiel Lopez-Rubio -- February 2011

% The mean of the scene is used as the original frame
MuImage = double(squeeze(shiftdim(model.Mu,2)));

% The smoothing approach is applied
SmoothingFilter = fspecial('gaussian',[3 3],0.5);
SmoothFrame=imfilter(MuImage,SmoothingFilter);

% The difference between the two images is obtained
dif = (MuImage - SmoothFrame).^2;

% A 0.01-winsorized mean is applied instead of the standard mean because
% the first measure is more robust and certain extreme values are removed
dif2 = reshape(dif,size(dif,1)*size(dif,2),model.Dimension);
dif3 = sort(dif2);
idx = round(length(dif3)*0.99);
for NdxDim=1:model.Dimension
    dif3(idx:end,NdxDim) = dif3(idx-1,NdxDim);
end

noise = mean(dif3);

