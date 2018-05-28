/* PART II: Update of the Stochastic Approximation Algorithm for background modelling from:

 Stochastic approximation for background modelling
 Ezequiel Lopez-Rubio and Rafael Marcos Luque-Baena
 Computer Vision and Image Understanding, DOI: 10.1016/j.cviu.2011.01.007

Example usage: See the matlab file test.m 

 Please note that the code is optimised for one gaussian and one uniform distribution, 
 so it does not work with another combination. 

Authors: R.M.Luque and Ezequiel Lopez-Rubio
Date: February 2011
*/

/*
Use the following command to compile this MEX file at the Matlab prompt:
mex -I. updateBM_MEX.cpp BasicMath.cpp
with tbb.lib and tbbmalloc_proxy.lib in the current folder. 
The compiled MEX file needs tbbmalloc_proxy.dll in the current folder in order to run.

*/

#include "mex.h"
#include "BasicMath.h"
#include <iostream>
#include "tbb/tbb.h"
//#include "tbb/tbbmalloc_proxy.h"



#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;
using namespace tbb;


/*---------------------------------------
 * Functions which return if the value is NaN or INF
 * They are needed because the Microsoft compiler (.NET 2008) does not include them as primitives 
 * (does not include the C99 especification)
 */
#ifdef _MSC_VER
//#ifndef isnan
bool isnan(double x) {
    return x != x;
}
#endif

#ifdef _MSC_VER
//#ifndef isinf
bool isinf(double x) {
    return ((x - x) != 0);
}
#endif
/* ------------------------------------------------------ */
#define REINITIALISATION_MODE 1

/* This is used for debugging porposes*/
#define DEBUG_MODE 0
#define MY_PIXEL 73389

/* Definition of pi number if it is not previously defined */
#ifndef M_PI
#define M_PI 3.141615
#endif



FILE * fich;

void PixelInitialisation(long NdxPixel,int Dimension,double *ptrMu,double *ptrMuFore,double *ptrC,double *ptrLogDetC,double *ptrNoise,double *ptrCounter,double *ptrPattern);

class AlmacenDatos {
    public:
        int Dimension;      //Antes era global
        
        long NumPixels;
        const int *DimPatterns;
        int NumImageRows,NumImageColumns;

        int NumCompUnif,NumCompGauss,NumComp,CurrentFrame,Z;
        double *ptrEpsilon,*ptrNumComp,*ptrNumCompUnif,*ptrNumCompGauss,*ptrPi;
        double *ptrCounter,*ptrNoise,*ptrMuFore,*ptrZ;
        double *ptrCurrentFrame,*pDataResp,*ptrOutput;
        double LearningRate,OneLessLearningRate;
        double LearningRateFore,OneLessLearningRateFore,PiFore,*ptrVectorProd,*ptrVectorDif;

        double *ptrMu,*ptrC,*ptrLogDetC,*ptrDen,*ptrMin,*ptrMax;
        int DimResp[3];
        double *ptrData;
        
        AlmacenDatos(mxArray **plhs, const mxArray **prhs){
            mxArray *Mu,*C,*LogDetC,*Min,*Max,*Den,*Counter,*MuFore;
            
             #if (DEBUG_MODE == 1) 
                /*Log variables*/
                char *fileName;
                mwSize loglen; 
                mxArray *Log;

                /* Getting the name of the log file*/
                Log = mxGetField(prhs[0],0,"Log");
                loglen = mxGetNumberOfElements(Log) + 1;
                fileName = malloc(loglen*sizeof(char));

                if (mxGetString(Log, fileName, loglen) != 0)
                    mexErrMsgTxt("Could not convert string data.");

                fich = OpenLog(fileName);
                fprintf(fich,"Beginning of the initialisation process\n");
            #endif

            /* Get input data */
            DimPatterns=mxGetDimensions(prhs[1]);
            NumImageRows = DimPatterns[0];
            NumImageColumns = DimPatterns[1];
            Dimension = DimPatterns[2];
            ptrData = (double *) mxGetData(prhs[1]);
            NumPixels = NumImageRows * NumImageColumns;

            /* Duplicate the model structure */
            plhs[0]=mxDuplicateArray(prhs[0]);

            /* Get the work variables */
            ptrNumCompGauss=mxGetPr(mxGetField(plhs[0],0,"NumCompGauss"));
            ptrNumCompUnif=mxGetPr(mxGetField(plhs[0],0,"NumCompUnif"));
            ptrNumComp=mxGetPr(mxGetField(plhs[0],0,"NumComp"));
            ptrEpsilon=mxGetPr(mxGetField(plhs[0],0,"Epsilon"));
            ptrPi=mxGetPr(mxGetField(plhs[0],0,"Pi"));

            Mu=mxGetField(plhs[0],0,"Mu");
            MuFore=mxGetField(plhs[0],0,"MuFore");
            C=mxGetField(plhs[0],0,"C");
            LogDetC=mxGetField(plhs[0],0,"LogDetC");
            Min=mxGetField(plhs[0],0,"Min");
            Max=mxGetField(plhs[0],0,"Max");
            Den=mxGetField(plhs[0],0,"Den");
            Counter=mxGetField(plhs[0],0,"Counter");

            NumCompGauss =(int)(*ptrNumCompGauss);
            NumCompUnif = (int)(*ptrNumCompUnif);
            NumComp=(int)(*ptrNumComp);

            /* Noise values for each dimension  */
            ptrNoise=mxGetPr(mxGetField(plhs[0],0,"Noise"));

            /* Work pointers */
            ptrMu = mxGetPr(Mu);
            ptrMuFore = mxGetPr(MuFore);
            ptrC = mxGetPr(C);
            ptrLogDetC = mxGetPr(LogDetC);
            ptrDen = mxGetPr(Den);
            ptrMax = mxGetPr(Max);
            ptrMin = mxGetPr(Min);
            ptrCounter = mxGetPr(Counter);

            #if (REINITIALISATION_MODE == 1)  
                ptrZ=mxGetPr(mxGetField(plhs[0],0,"Z"));
                Z = (int)(*ptrZ);	
                #if (DEBUG_MODE == 1) 
                    fprintf(fich,"Z value for pixel reinitialisation: %d\n", Z);
                #endif
            #endif
                        
            /* Create a matrix for the return argument */
            DimResp[0]=DimPatterns[0];
            DimResp[1]=DimPatterns[1];
            DimResp[2] = NumComp;
            plhs[1] = mxCreateNumericArray(2,DimPatterns,mxDOUBLE_CLASS, mxREAL);
            plhs[2] = mxCreateNumericArray(3,DimResp,mxDOUBLE_CLASS, mxREAL);

            /* Assign pointers to the outputs */
            ptrOutput = (double*) mxGetData(plhs[1]);
            pDataResp = (double*) mxGetData(plhs[2]);

            /* Update the current frame */
            ptrCurrentFrame=mxGetPr(mxGetField(plhs[0],0,"CurrentFrame")); 
            (*ptrCurrentFrame) = (*ptrCurrentFrame) + 1.0;
            CurrentFrame=(int)(*ptrCurrentFrame);

            #if (DEBUG_MODE == 1)  
                if (CurrentFrame == 1) {
                    fprintf(fich,"Beginning of the update process\n");
                    fprintf(fich,"Sequence noise: ");
                    RecordMatrixLog(fich,ptrNoise,1,Dimension); 
                }
                fprintf(fich,"Current frame nº: %d\n", CurrentFrame);	
            #endif

            /* Load the learning rate */
            LearningRate=(*ptrEpsilon);
            OneLessLearningRate=1.0-LearningRate;
            LearningRateFore=(*ptrEpsilon);
            OneLessLearningRateFore=1.0-LearningRateFore;
        }
};

class AplicadorParallel {
    public:
    AlmacenDatos *const mis_Datos;
    
    void operator()( const blocked_range<size_t>& r ) const {
        const AlmacenDatos *Datos = mis_Datos;
        double *aux_Pattern,*aux_ptrPi,*aux_ptrTempVector;
        double *aux_ptrCounter,*aux_ptrMuFore;
        double *aux_pDataResp,*aux_ptrOutput;
        double aux_GaussianResponsibilities,aux_SumResponsibilities,aux_AntPi,aux_CoefOld,aux_CoefNew,aux_CoefOldFore,aux_CoefNewFore;
        double aux_PiFore,*aux_ptrVectorProd,*aux_ptrVectorDif,*aux_ptrResponsibilities;

        double aux_MyLogDensity;
        double aux_DistMahal;
        double *aux_ptrMu,*aux_ptrC,*aux_ptrLogDetC;
        double *aux_ptrData;
        double *aux_ptrSigma;
        Map<MatrixXd> *aux_Sigma;
        LLT<MatrixXd> *aux_MyLLT;
        Map<VectorXd> *aux_VectorDif;
        MatrixXd *aux_L;
        int aux_NdxDim,aux_NdxComp,aux_NdxPixel;
        
        /* Allocate dynamic data */
        aux_Pattern=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrVectorProd=(double *) malloc(Datos->Dimension*Datos->Dimension*sizeof(double));
		aux_ptrVectorDif=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrTempVector=(double *) malloc(Datos->Dimension*sizeof(double));
		aux_ptrResponsibilities=(double *) malloc(Datos->NumComp*sizeof(double));
		aux_ptrSigma=(double *) malloc(Datos->Dimension*Datos->Dimension*sizeof(double));
		aux_Sigma=new Map<MatrixXd>(aux_ptrSigma,Datos->Dimension,Datos->Dimension);
		aux_VectorDif=new Map<VectorXd>(aux_ptrVectorDif,Datos->Dimension);
		aux_MyLLT=new LLT<MatrixXd>(Datos->Dimension);
		aux_L=new MatrixXd(Datos->Dimension,Datos->Dimension);
        
	    /* Pointers asociated to the mixture components are incremented */
        aux_ptrPi=Datos->ptrPi+r.begin()*Datos->NumComp;
        aux_ptrMu=Datos->ptrMu+r.begin()*Datos->NumCompGauss*Datos->Dimension;
        aux_ptrMuFore=Datos->ptrMuFore+r.begin()*Datos->NumCompGauss*Datos->Dimension;
        aux_ptrC=Datos->ptrC+r.begin()*Datos->NumCompGauss*Datos->Dimension*Datos->Dimension;
        aux_ptrLogDetC=Datos->ptrLogDetC+r.begin()*Datos->NumCompGauss;

        /* Global pointers are incremented */
        aux_ptrData=Datos->ptrData+r.begin();
        aux_pDataResp=Datos->pDataResp+r.begin();
        aux_ptrOutput=Datos->ptrOutput+r.begin();
        aux_ptrCounter=Datos->ptrCounter+r.begin();
		

		for (aux_NdxPixel=r.begin(); aux_NdxPixel!=r.end(); ++aux_NdxPixel)
		{
			/* MATLAB code: aux_Pattern = Patterns(:,NdxPattern); */
			for(aux_NdxDim=0;aux_NdxDim<Datos->Dimension;aux_NdxDim++)
			{
				aux_Pattern[aux_NdxDim]=aux_ptrData[aux_NdxDim*Datos->NumPixels];
			}
		
			/* The pixel is reinitialised if it belongs to the foreground too much time */
			#if (REINITIALISATION_MODE == 1)  
			if (*aux_ptrCounter > Datos->Z) PixelInitialisation(aux_NdxDim,Datos->Dimension,aux_ptrMu,aux_ptrMuFore,aux_ptrC,aux_ptrLogDetC,Datos->ptrNoise,aux_ptrCounter,aux_Pattern);
			#endif

			/* ------------------------------------------------------------------------------ */ 
			/* Start of the code to compute the responsibilities */
			/* ------------------------------------------------------------------------------ */

			/* Responsibilities of the Gaussian mixture components */
			aux_SumResponsibilities=0.0;
			for(aux_NdxComp=0;aux_NdxComp<Datos->NumCompGauss;aux_NdxComp++)
			{
				/* MATLAB code: Differences=aux_Pattern-Model.Mu{NdxCompGauss} */
				Difference(aux_Pattern,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrVectorDif,Datos->Dimension);

				/* Compute the determinant of C+Psi and the squared Mahalanobis distance */
				/* MATLAB code: aux_DistMahal=Differences'*inv(Model.C{NdxCompGauss}+diag(Model.Noise))*Differences; */
				/* aux_Sigma=C+Psi; */
				memcpy(aux_ptrSigma,aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,Datos->Dimension*Datos->Dimension*sizeof(double));
				SumMatrixDiagonal(aux_ptrSigma,Datos->ptrNoise,NULL,Datos->Dimension); /* Add the noise to the diagonal of the covariance matrix */
				/* aux_Sigma=aux_L*aux_L'; (Cholesky decomposition)*/
				aux_MyLLT->compute(*aux_Sigma);
				(*aux_L)= aux_MyLLT->matrixL();
				/* log(det(aux_Sigma))=2.0*log(det(aux_L)); */
				aux_ptrLogDetC[aux_NdxComp]=2.0*log(aux_L->diagonal().prod());
				/* b=inv(aux_L)*Differences; <=> b=aux_L\Differences; */
				aux_L->triangularView<Lower>().solveInPlace(*aux_VectorDif);
				/* aux_DistMahal=b'*b; */
				aux_DistMahal=aux_VectorDif->squaredNorm();		

				/* aux_MyLogDensity=-0.5*Datos->Dimension*log(2*M_PI)-0.5*(*aux_ptrLogDetC)-0.5*aux_DistMahal; */
				aux_MyLogDensity=-0.918938533204673*Datos->Dimension-0.5*aux_ptrLogDetC[aux_NdxComp]-0.5*aux_DistMahal;

				/* MATLAB code: p(t sub n | i);  aux_ptrResponsibilities[aux_NdxComp]=Model.Pi(NdxCompGauss)*exp(aux_MyLogDensity); */
				aux_ptrResponsibilities[aux_NdxComp] = aux_ptrPi[aux_NdxComp]*exp(aux_MyLogDensity);

				/* Discard NaN and INF values for responsabilities  */
				if (isnan(aux_ptrResponsibilities[aux_NdxComp]) || isinf(aux_ptrResponsibilities[aux_NdxComp]))
				{
					aux_ptrResponsibilities[aux_NdxComp] = 0.0;
				}

				/* Accumulate the resposibility for later normalization */
				aux_SumResponsibilities+=aux_ptrResponsibilities[aux_NdxComp];
			}

			/* Responsibilities of the uniform mixture components */
			for(aux_NdxComp=Datos->NumCompGauss;aux_NdxComp<Datos->NumComp;aux_NdxComp++)
			{
				aux_ptrResponsibilities[aux_NdxComp]=aux_ptrPi[aux_NdxComp]*Datos->ptrDen[aux_NdxComp-Datos->NumCompGauss];

				/* Accumulate the resposibility for later normalization */
				aux_SumResponsibilities+=aux_ptrResponsibilities[aux_NdxComp];
			}
			    
			/* Normalize the responsabilities and return them. An extremely low value is added to the denominator 
			  * in order to have value higher than 0 */
			aux_SumResponsibilities+=0.000000001;
			aux_GaussianResponsibilities=0.0;
			for(aux_NdxComp=0;aux_NdxComp<Datos->NumComp;aux_NdxComp++)
			{
				aux_ptrResponsibilities[aux_NdxComp]/=aux_SumResponsibilities;
				aux_pDataResp[aux_NdxComp*Datos->NumPixels]=aux_ptrResponsibilities[aux_NdxComp];
				if (aux_NdxComp<Datos->NumCompGauss)
				{
					aux_GaussianResponsibilities+=aux_ptrResponsibilities[aux_NdxComp];
				}
			}

			/* The output is the sum of the responsibilities of the Gaussian distributions (between 0 and 1). 
			* The higher the value, the more probability to belong to the background of the sequence. 
			* It is considered that the Gaussian distributions model the background of the sequence whereas the 
			* the uniform distribution deal with the foreground part */
			*(aux_ptrOutput) = aux_GaussianResponsibilities;

			/* The counter is incremented if the pixel belongs to the foreground */
			if (aux_GaussianResponsibilities < 0.5) (*aux_ptrCounter)+=1;
			else (*aux_ptrCounter)=0;

			/* ------------------------------------------------------------------------------ */
			/* End of the code to compute the responsibilities */
			/* ------------------------------------------------------------------------------ */
				
			#if (DEBUG_MODE == 1)
				if (aux_NdxPixel==MY_PIXEL) {
					fprintf(fich,"Pixel nº %d\n",aux_NdxPixel);	
					RecordMatrixLog(fich,aux_Pattern,1,Datos->Dimension);
					fprintf(fich,"Responsabilities: ");
					RecordMatrixLog(fich,aux_ptrResponsibilities,1,Datos->NumComp);
				}
			#endif

			/* ------------------------------------------------------------------------------ */ 
			/* Start of the code to update the model */
			/* ------------------------------------------------------------------------------ */
			
			/* Update of the parameters of the Gaussian distributions */ 
			for(aux_NdxComp=0;aux_NdxComp<Datos->NumCompGauss;aux_NdxComp++)
			{
				aux_AntPi=aux_ptrPi[aux_NdxComp];
				aux_ptrPi[aux_NdxComp]=Datos->OneLessLearningRate*aux_ptrPi[aux_NdxComp] + Datos->LearningRate*aux_ptrResponsibilities[aux_NdxComp];

				aux_CoefOld=(Datos->OneLessLearningRate*aux_AntPi)/aux_ptrPi[aux_NdxComp];
				aux_CoefNew=(Datos->LearningRate*aux_ptrResponsibilities[aux_NdxComp])/aux_ptrPi[aux_NdxComp];

				aux_PiFore = 1.0 - Datos->OneLessLearningRateFore*aux_ptrPi[aux_NdxComp] + Datos->LearningRateFore*aux_ptrResponsibilities[aux_NdxComp];
				aux_CoefOldFore=(Datos->OneLessLearningRateFore*(1.0-aux_AntPi))/(aux_PiFore);
				aux_CoefNewFore=(Datos->LearningRateFore*(1.0-aux_ptrResponsibilities[aux_NdxComp]))/(aux_PiFore);

				#if (DEBUG_MODE == 1)
					if (aux_NdxPixel==MY_PIXEL) fprintf(fich,"aux_CoefOld: %f aux_CoefNew: %f\n",aux_CoefOld,aux_CoefNew);
				#endif

				/* MATLAB code: Model.Mu{aux_NdxComp} = (1-Model.Epsilon)*Model.Mu{aux_NdxComp} + ...
					* Model.Epsilon*R*Patterns(:,NdxPattern);  */
				ScalarMatrixProduct(aux_CoefNew,aux_Pattern,aux_ptrTempVector,Datos->Dimension,1);
				ScalarMatrixProduct(aux_CoefOld,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrMu+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);
				MatrixSum(aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrTempVector,aux_ptrMu+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);

				/* MATLAB code: Model.MuFore{aux_NdxComp} = (1-Model.Epsilon)*Model.MuFore{aux_NdxComp} + ... 
						   * Model.Epsilon*R_fore*Patterns(:,NdxPattern);  */
				ScalarMatrixProduct(aux_CoefNewFore,aux_Pattern,aux_ptrTempVector,Datos->Dimension,1);
				ScalarMatrixProduct(aux_CoefOldFore,aux_ptrMuFore+aux_NdxComp*Datos->Dimension,aux_ptrMuFore+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);
				MatrixSum(aux_ptrMuFore+aux_NdxComp*Datos->Dimension,aux_ptrTempVector,aux_ptrMuFore+aux_NdxComp*Datos->Dimension,Datos->Dimension,1);

				/* MATLAB code: aux_VectorDif=Patterns(:,NdxPattern) - Model.Mu{NdxCompGauss}; */
				Difference(aux_Pattern,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrVectorDif,Datos->Dimension);

				/* MATLAB code: Model.C{aux_NdxComp} = (1-Model.Epsilon)*Model.C{aux_NdxComp} + ...
					* Model.Epsilon*R*Difference*Difference'; */
				Difference(aux_Pattern,aux_ptrMu+aux_NdxComp*Datos->Dimension,aux_ptrVectorDif,Datos->Dimension);
				MatrixProduct(aux_ptrVectorDif,aux_ptrVectorDif,aux_ptrVectorProd,Datos->Dimension,1,Datos->Dimension);
				ScalarMatrixProduct(aux_CoefNew,aux_ptrVectorProd,aux_ptrVectorProd,Datos->Dimension,Datos->Dimension);
				ScalarMatrixProduct(aux_CoefOld,aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,Datos->Dimension,Datos->Dimension);
				MatrixSum(aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,aux_ptrVectorProd,aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,Datos->Dimension,Datos->Dimension);

				/* The inverse and the logarithm of the determinant of the covariance matrix are computed */
				//memcpy(aux_ptrSigma,aux_ptrC+aux_NdxComp*Datos->Dimension*Datos->Dimension,Datos->Dimension*Datos->Dimension*sizeof(double));
				//SumMatrixDiagonal(aux_ptrSigma,Datos->ptrNoise,NULL,Datos->Dimension); /* Add the noise to the diagonal of the covariance matrix */
				//MyLU->compute(*aux_Sigma);
				//(*MyInvC)=MyLU->inverse();
				//aux_ptrLogDetC[aux_NdxComp] = log(MyLU->determinant());
				//memcpy(ptrInvC,TempInvC,Datos->Dimension*Datos->Dimension*sizeof(double));

			}


			/* ------------------------------------------------------------------------------ */
			/* End of the code to update the model */
			/* ------------------------------------------------------------------------------ */

			
			/* Record the relevant pixel information to the log */
			#if (DEBUG_MODE == 1)
			if (aux_NdxPixel==MY_PIXEL) {
				fprintf(fich,"Pi: %f Counter: %f\n",*aux_ptrPi,*aux_ptrCounter);
				fprintf(fich,"Mu\n");
				RecordMatrixLog(fich,aux_ptrMu,1,Datos->Dimension); 
				fprintf(fich,"C\n");
				RecordMatrixLog(fich,aux_ptrC,Datos->Dimension,Datos->Dimension); 
				fprintf(fich,"InvC\n");
				RecordMatrixLog(fich,ptrInvC,Datos->Dimension,Datos->Dimension);
				fprintf(fich,"log(DetC): %f\n",*aux_ptrLogDetC); 
				fprintf(fich,"MuFore\n");
				RecordMatrixLog(fich,aux_ptrMuFore,1,Datos->Dimension); 
			}
			#endif

			/* Pointers asociated to the mixture components are incremented */
			aux_ptrPi+=Datos->NumComp;
			aux_ptrMu+=Datos->NumCompGauss*Datos->Dimension;
			aux_ptrMuFore+=Datos->NumCompGauss*Datos->Dimension;
			aux_ptrC+=Datos->NumCompGauss*Datos->Dimension*Datos->Dimension;
			aux_ptrLogDetC+=Datos->NumCompGauss;
			
			/* Global pointers are incremented */
			aux_ptrData++;
			aux_pDataResp++;
			aux_ptrOutput++;
			aux_ptrCounter++;
		}

			
		#if (DEBUG_MODE == 1) 
			/* Close the log file */
			CloseLog(fich);
		#endif
             
         /* Release pointers */       
        free(aux_Pattern);
		free(aux_ptrVectorProd);
		free(aux_ptrVectorDif);
		free(aux_ptrTempVector);
		free(aux_ptrResponsibilities);
		free(aux_ptrSigma);
		delete aux_Sigma;
		delete aux_MyLLT;
		delete aux_VectorDif;
		delete aux_L;
    }
    
    AplicadorParallel(AlmacenDatos *const aux_Datos) : mis_Datos(aux_Datos)
    {
    }
        
    ~AplicadorParallel(){
        

    }
};

void mexFunction( int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs )     
{
	const int *DimPatterns;
	int NumImageRows,NumImageColumns;
	long NumPixels;
	/* Get input data */
    DimPatterns=mxGetDimensions(prhs[1]);
	NumImageRows = DimPatterns[0];
	NumImageColumns = DimPatterns[1];
	NumPixels = NumImageRows * NumImageColumns;
    
    
	/* For each one of the image pixels */
	//for (aux_NdxPixel=0;aux_NdxPixel<NumPixels;aux_NdxPixel++)
    AlmacenDatos *const Datos = new AlmacenDatos(plhs, prhs);
    Eigen::initParallel();
	
	/*int n = task_scheduler_init::default_num_threads();
    std::printf("n=%d\n",n);

	task_scheduler_init(2);*/
	parallel_for( blocked_range<size_t>(0,NumPixels), AplicadorParallel(Datos));
	delete Datos;
}
 
/**********************************************************************************************
 * Function to reinitialise a pixel which has exceeded the Z value. Z is the maximun number of consecutive 
 * frames in which a pixel belongs to the foreground class.
 **********************************************************************************************/
void PixelInitialisation(long NdxPixel,int Dimension,double *ptrMu,double *ptrMuFore,double *ptrC,double *ptrLogDetC,double *ptrNoise,double *ptrCounter,double *ptrPattern) 
{
	double tmpLogDetC;
	double *temp;
	int NdxDim;

	temp=(double *)malloc(Dimension*sizeof(double));

	/* The counter for this pixel is initialised*/
	*ptrCounter = 0;

	/* Swap between Mu and MuFore*/
	memcpy(temp,ptrMu,Dimension*sizeof(double));
	memcpy(ptrMu,ptrMuFore,Dimension*sizeof(double));
	memcpy(ptrMuFore,temp,Dimension*sizeof(double));

	#if (DEBUG_MODE == 1)
		if (NdxPixel==MY_PIXEL) {
			fprintf(fich,"REINITIALISATION OF THE PIXEL\n");
			fprintf(fich,"ptrMu:");
			RecordMatrixLog(fich,ptrMu,1,Dimension); 
			fprintf(fich,"ptrMuFore:");
			RecordMatrixLog(fich,ptrMuFore,1,Dimension); 
			fprintf(fich,"--------------------------\n");
		}
	#endif

	/* Initialisation of the covariance matrix in combination with the noise */
	memset(ptrC,0,Dimension*Dimension*sizeof(double));
	/* Add the noise to the diagonal of the covariance matrix */
	SumMatrixDiagonal(ptrC,ptrNoise,NULL,Dimension);

    /* Compute the logarithm of the determinant of the covariance matrix */
	tmpLogDetC=0.0;
	for(NdxDim=0;NdxDim<Dimension;NdxDim++)
	{
		tmpLogDetC+=log(ptrNoise[NdxDim]);
	}
	*ptrLogDetC = tmpLogDetC;
	
	/* Release pointers */
	free(temp);
}
