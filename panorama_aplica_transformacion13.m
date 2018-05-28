
function [model fallo]= panorama_aplica_transformacion13(imActual,model, transformacion, tau)

    %%Inicializacion de vbles que se calcularon en extrae_informacion
    xmin=transformacion.xmin;
    xmax=transformacion.xmax;
    ymin=transformacion.ymin;
    ymax=transformacion.ymax;
    nchannels = size(imActual,3);
    tamInicial=transformacion.tamInicial;   %Numero de pixeles extendidos [Alto x Ancho]
    objgridX=transformacion.objgridX;
    objgridY=transformacion.objgridY;
    gridsize=transformacion.gridsize;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    %%Calculamos el R a partir de C y de Mu
    R=reshape(model.C + multiprod(model.Mu,model.Mu,[1 0],[0 1]), ...
        [nchannels*nchannels size(model.Mu,2) size(model.Mu,3)]);
    
    fallo=false;    
    
    TamBigImages =  [ymax-ymin+1, xmax-xmin+1]; %Tamaño de las dimensiones del marco del frame actual
    PosImActual_y = -ymin+2:-ymin+size(imActual,1)+1;                
    PosImActual_x = -xmin+2:-xmin+size(imActual,2)+1;
  
    [TransMu_nD TransMuFore_nD TransCounter TransR_nD TransPi_nD Corona] = ...
        aplica_transformacion6_MEX(objgridX,objgridY, model.Mu,...
        model.MuFore, R, model.Pi, model.Counter,...
        tamInicial(2), tamInicial(1), size(objgridX,1), size(model.Mu,2),...
        size(model.Mu,3), nchannels);

    TransMu_nD=reshape(TransMu_nD(:), nchannels, gridsize(1),gridsize(2));
    TransMuFore_nD=reshape(TransMuFore_nD(:),nchannels, gridsize(1),gridsize(2));
    TransCounter=reshape(TransCounter(:),gridsize(1),gridsize(2));
    TransPi_nD=reshape(TransPi_nD(:),2, gridsize(1),gridsize(2));
    Corona=reshape(Corona(:),gridsize(1),gridsize(2));    
    TransR_nD=reshape(TransR_nD,[nchannels nchannels gridsize]);
    
    %%Todos los Big... almacenan los resultados finales
    %%los inicializamos a los valores del frame anterior transformado
    PosImActual_y=PosImActual_y-1;
    PosImActual_x=PosImActual_x-1;
    
    BigImageMuFore = TransMuFore_nD;
    BigImageCounter = TransCounter;
    BigImagePi = TransPi_nD;
    
    BigImageMu = zeros([nchannels TamBigImages]);
    %En BigImageMu colocamos el frame actual:
    BigImageMu(:,PosImActual_y,PosImActual_x) = shiftdim(imActual,2);
    
    aux=BigImageMu;
    Pixeles_En_Frame_Act=zeros(TamBigImages);
    Pixeles_En_Frame_Act(PosImActual_y,PosImActual_x)= 1;
    
    %Comprobamos que la tranformacion se ha realizado bien en .mex
    if sum(size(Pixeles_En_Frame_Act)-size(Corona))~=0    %Se trata de evitar errores debidos a malos emparejamientos=> no se actualiza ese frame
        'Fallo en actualizacion (3): se ignora frame'     
        fallo = true;
        return;
    end  
    
    %En BigImageMu machacamos la parte de Mu anterior que cae en el frame actual:
    idxFrmAnt=Pixeles_En_Frame_Act & Corona(:,:)<=0.5;   
    BigImageMu(:,idxFrmAnt)= TransMu_nD(:,idxFrmAnt);
  %%Calculamos el C transformado a partir de R y de Mu transformados
    C_aux=TransR_nD-multiprod(TransMu_nD,TransMu_nD,[1 0],[0 1]);
    BigImageC=C_aux;    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%Actualizamos los pixeles correspondientes al borde extendido:  

    %%Calculamos las responsabilities
    DifferencesMatrix=aux-TransMu_nD;
    DistMahalMatrix=zeros(size(C_aux,3),size(C_aux,4));
    LogDetCMatrix=zeros(size(C_aux,3),size(C_aux,4));
    
    Pixeles_En_Borde_matrix=Pixeles_En_Frame_Act & Corona(:,:)>0.5;
    indices= find(Pixeles_En_Borde_matrix);
    diagonal=diag(model.Noise);  
    for idx = indices'      
        Sigma =C_aux(:,:,idx)+diagonal;
        DistMahalMatrix(idx)=DifferencesMatrix(:,idx)'*(Sigma\DifferencesMatrix(:,idx));
        LogDetCMatrix(idx)=log(det(Sigma));
    end
    MyLogDensityMatrix=-0.918938533204673*model.Dimension-0.5*LogDetCMatrix-0.5*DistMahalMatrix;
    APriori=shiftdim(TransPi_nD(1,:,:),1);
    Numerator=APriori.*exp(MyLogDensityMatrix);
    ResponsibilitiesMatrix=Numerator./(Numerator+(1-APriori));
    
    %%Zona nueva => Calculamos si el color es similar al del borde anterior
    %%extendido

    idxCopiarModeloBorde=ResponsibilitiesMatrix>tau & Pixeles_En_Borde_matrix;
    BigImageMu(:, idxCopiarModeloBorde)=TransMu_nD(:, idxCopiarModeloBorde);
    
    %%Zona nueva => Si el color es diferente al del borde anterior
    %%extendido, ponemos valores por defecto
    idxResetModeloBorde=ResponsibilitiesMatrix<=tau & Pixeles_En_Borde_matrix;
    BigImageCounter(idxResetModeloBorde)=0;
    BigImagePi(:,idxResetModeloBorde)= 0.5;
    
    BigImageC(:,:,idxResetModeloBorde)=0;
    for ch = 1:nchannels
        BigImageC(ch,ch,idxResetModeloBorde)=model.Noise(ch);
    end        
    BigImageMuFore(:,idxResetModeloBorde)=0.5;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Copiamos resultados finales al modelo
    model.Mu = BigImageMu(:, PosImActual_y,PosImActual_x);
    model.MuFore = BigImageMuFore(:, PosImActual_y,PosImActual_x);    
    model.C = BigImageC(:,:, PosImActual_y,PosImActual_x);    
    model.Pi = BigImagePi(:,PosImActual_y,PosImActual_x);    
    model.Counter = BigImageCounter( PosImActual_y,PosImActual_x);
end