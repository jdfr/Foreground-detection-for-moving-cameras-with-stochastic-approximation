
function [transformacion fallo]= panorama_extrae_transformacion8(imActual,model, minHessian, ransacReproj, coefReplicaBorde)

    %Preallocate cell arrays
    I      = cell(2,1);
    
    valido = false;
    fallo = false;
    cont = 0;
    while not(valido) && (cont < 10)
        %Controla le numero de ejecuciones del bucle, si supera un limite
        %sale devolviendo fallo
        cont=cont+1;
        
        %Read file
        I{1} = uint8(shiftdim(imActual,2));     %0 a 255 uint8
        %Calculate SURF descriptor

        %Read file
        I{2} = uint8(model.Mu*255);
        tamInicial = ceil(size(I{2})*coefReplicaBorde);
        I{2} = padarray(I{2},[0 tamInicial(2) tamInicial(3)],'replicate','both');
        %Calculate SURF descriptor 
        H = extrae_transformacion_BF2_MEX(I{1}, I{2}, 1, minHessian, ...
            ransacReproj, size(I{1},2), size(I{1},3), size(I{2},2), size(I{2},3)); %Usando BruteForce Matcher, mas rapido
        
        %Obtenemos los extremos de la imagen transformada para saber como
        %de grande tendra que ser BigImage
                
        gridxmax = size(I{2},3);
        gridymax = size(I{2},2);

        transformedCorners = H \ [1 1 1;gridxmax 1 1; gridymax 1 1;gridxmax gridymax  1]';
        transformedCorners(1,:)=transformedCorners(1,:)./transformedCorners(3,:);
        transformedCorners(2,:)=transformedCorners(2,:)./transformedCorners(3,:);
        
        %Update the maximum and minimum coordinates
        xmax=max(transformedCorners(1,:));
        ymax=max(transformedCorners(2,:));
        xmin=min(transformedCorners(1,:));
        ymin=min(transformedCorners(2,:));
    
        xmin = ceil(xmin);
        xmax = floor(xmax);
        ymin = ceil(ymin);
        ymax = floor(ymax);
      
        %Controlamos que no haya habido una transformacion erronea

        X=(xmax-xmin);  
        Y=(ymax-ymin);
        y=size(imActual,1);
        x=size(imActual,2);

        crecimientoErroneo =((0.6*x>X)||(2*x<X)||(0.6*y>Y)||(2*y<Y));            

        valido = not(crecimientoErroneo) && (-ymin+2)>1 && (-xmin+2)>1;
        if not(valido)
            'Minor error' %%Fallo leve, en extracción
        end
    end
    
    if not(valido)
        fallo = true;   %%Fallo grave
        transformacion = [];
        return;
    end
        
    transformacion.xmin=xmin;
    transformacion.xmax=xmax;
    transformacion.ymin=ymin;
    transformacion.ymax=ymax;
    transformacion.tamInicial(1)=tamInicial(2);
    transformacion.tamInicial(2)=tamInicial(3);
    
    [biggridX biggridY] = meshgrid(xmin:xmax,ymin:ymax);
    transformacion.gridsize = size(biggridX);
    transformedgrid = single([biggridX(:) biggridY(:) ones(numel(biggridX),1)]*H');
    transformacion.objgridX = transformedgrid(:,1) ./ transformedgrid(:,3);
    transformacion.objgridY = transformedgrid(:,2) ./ transformedgrid(:,3);
    
end