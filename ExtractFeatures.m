function [FeatureFrame]=ExtractFeatures(VideoFrame,SelectedFeatures)

NumRowsImg=size(VideoFrame,1);
NumColsImg=size(VideoFrame,2);
NumFeatures=numel(SelectedFeatures);
FeatureFrame=zeros(NumRowsImg,NumColsImg,NumFeatures);
for NdxFeature=1:NumFeatures
    MyFeature=SelectedFeatures(NdxFeature);
    switch(MyFeature)
        case {1,2,3}
            % Red, green and blue channels
            FeatureFrame(:,:,NdxFeature)=VideoFrame(:,:,MyFeature)/255;
        case {4,5,6}
            % Normalized red, green and blue channels
            SumFrame=sum(VideoFrame,3);
            SumFrame(SumFrame==0)=1;
            FeatureFrame(:,:,NdxFeature)=VideoFrame(:,:,MyFeature-3)./SumFrame;
        case {7,8,9,10,11,12}
            % Haar-like features considered in Han & Davis (2012)
            SumFrame=sum(VideoFrame,3);
            MyFilter=HFilter(MyFeature-6);
            FeatureFrame(:,:,NdxFeature)=imfilter(SumFrame,MyFilter)/(3*255*sum(abs(MyFilter(:))));
        case 13
            % Gradient in the horizontal direction, considered in Han & Davis (2012)
            SobelFilter=[-1 0 1;-2 0 2;-1 0 1];
            SumFrame=sum(VideoFrame,3);
            FeatureFrame(:,:,NdxFeature)=imfilter(SumFrame,SobelFilter)/(3*255*sum(abs(SobelFilter(:))));
        case 14
            % Gradient in the vertical direction, considered in Han & Davis (2012)
            SobelFilter=[-1 0 1;-2 0 2;-1 0 1]';
            SumFrame=sum(VideoFrame,3);
            FeatureFrame(:,:,NdxFeature)=imfilter(SumFrame,SobelFilter)/(3*255*sum(abs(SobelFilter(:))));     
        case 15
            % Red channel of the current pixel, normalized with the current pixel 
            % and the pixel immediately to the left of the current one
            ShiftedFrame=circshift(VideoFrame,[1 0 0]);
            SumFrame=sum(VideoFrame+ShiftedFrame,3);
            SumFrame(SumFrame==0)=1;
            FeatureFrame(:,:,NdxFeature)=6*VideoFrame(:,:,1)./SumFrame;     
        case 16
            % Green channel of the pixel immediately to the lower right of the
            % current one, normalized with the current pixel
            ShiftedFrame=circshift(VideoFrame,[-1 -1 0]);
            SumFrame=sum(VideoFrame,3);
            SumFrame(SumFrame==0)=1;
            FeatureFrame(:,:,NdxFeature)=3*ShiftedFrame(:,:,2)./SumFrame; 
        case {17,18,19}
            % Red, green and blue channels, median filtered
            FeatureFrame(:,:,NdxFeature)=double(medfilt2(uint8(squeeze(VideoFrame(:,:,MyFeature-16))),[5 5]))/255;
        case {20,21,22}
            % Normalized red, green and blue channels, median filtered
            SumFrame=sum(VideoFrame,3);
            SumFrame(SumFrame==0)=1;
            NormFrame=squeeze(VideoFrame(:,:,MyFeature-19))./SumFrame;            
            FeatureFrame(:,:,NdxFeature)=double(medfilt2(uint16(32768*NormFrame),[5 5]))/32768;
        case 23
            % Tiny Haar-like feature (I)
            MyFilter=[1 0 1;1 0 1;1 0 1];
            SumFrame=sum(VideoFrame,3);
            FeatureFrame(:,:,NdxFeature)=imfilter(SumFrame,MyFilter)/(3*6*255);             
        case 24
            % Tiny Haar-like feature (II)
            MyFilter=[1 1 1;1 0 1;1 1 1];
            SumFrame=sum(VideoFrame,3);
            FeatureFrame(:,:,NdxFeature)=imfilter(SumFrame,MyFilter)/(3*8*255);                             
    end
end
end

% Filters corresponding to the Haar-like features considered in:
% Han, B. and Davis, L.S. (2012). Density-Based Multifeature Background
% Subtraction with Support Vector Machine. IEEE Transactions on Pattern
% Analysis and Machine Intelligence 34(5), 1017-1023.
function [Filter]=HFilter(NdxFilter)

H=cell(6,1);
H{1}=[1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1];
H{2}=H{1}';
    
H{3}=[1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5; ...
    0 0 0 0 0.5 1 1 1 1; ...
    0 0 0 0 0.5 1 1 1 1; ...
    0 0 0 0 0.5 1 1 1 1; ...
    0 0 0 0 0.5 1 1 1 1];

H{4}=[1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0; ...
    1 1 1 1 0.5 0 0 0 0];

H{5}=H{4}';

H{6}=[1 1 1 1 1 1 1 1 1; ...
    1 1 1 1 1 1 1 1 1; ...
    1 1 1 1 1 1 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 0 0 0 1 1 1; ...
    1 1 1 1 1 1 1 1 1; ...
    1 1 1 1 1 1 1 1 1; ...
    1 1 1 1 1 1 1 1 1];

Filter=H{NdxFilter};
    
end