clear all

%woman
idx = 1;
VideoFileSpec{idx}='woman/frames/scene%05d.png';
deltaFrame{idx}=0;
numFrames{idx}=557;

%badminton
idx = 2;
VideoFileSpec{idx}='badminton/frames/scene%05d.png';
deltaFrame{idx}=0;
numFrames{idx}=1150;

Features{1}=[3 20 21 22];  
Features{2}=[5 19 20 21 22];
Features{3}=[19 20 22];  

%%Choose the sequence number: 1 or 2
NdxVideo=1; 

%%Choose the feature set: 1, 2 or 3
FeatureIdx=3;

%%Set Epsilon value: 0.002, 0.01, 0.02 or 0.03
Epsilon=0.03;

%%Set Tau value: 0.999, 0.9995 or 0.9999
tau=0.999;

simulaSecuenciaParanoramaBM4(VideoFileSpec{NdxVideo}, deltaFrame{NdxVideo}, numFrames{NdxVideo},...
    Features{FeatureIdx}, Epsilon, 400, 3, 0.2, 0.09, tau);





