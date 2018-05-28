/*PART I: Initialisation of the Stochastic Approximation Algorithm for background modelling from:

 F. J. Lopez Rubio and E. Lopez-Rubio. Features for stochastic approximation based foreground detection. Computer Vision and Image Understanding. ISSN: 1077-3142.

http://dx.doi.org/10.1016/j.cviu.2014.12.007
*/
/*
Use the following command to compile this MEX file at the Matlab prompt:
mex -IEigen initializeBM_MEX.cpp BasicMath.cpp
This works for Windows and Linux.
*/


#include "mex.h"
#include "BasicMath.h"


#include <Eigen/Eigen>

using namespace Eigen;

/* Debug mode is activated if the variable is 1*/
#define DEBUG_MODE 0

/* For debugging porpuses a specific pixel is selected:
  * Calculate the pixel (x,y) with image NumPixels (M,N)
   MI_PIXEL = (x-1)*M + (y-1)	
   Ej: Pixel (153,430) y NumPixels (480,640)
   MI_PIXEL = (153-1)*480 + (430 - 1) = 73389
 */
#define MI_PIXEL 73389


void GetPositionData(double * data, double * output, int Dimension, int NumPixels, int NumFrames);
void MyMean(double *ptrMean,double *data,int Dimension,int NumPatterns);
void MyCov(double *ptrCov,double *data,int Dimension,int NumPatterns);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )     
{
	int Dimension,NumFrames,NumImageRows,NumImageColumns,NumCompUnif,NumCompGauss,NumComp; 
	long NumPixels;
	mxArray *Mu,*C,*Min,*Max,*Den,*MuFore;
	double *data,*ptrNumComp,*ptrNumCompUnif,*ptrNumCompGauss,*ptrPi;
	double *ptrCurrentFrame;
	double *ptrMu,*ptrMuFore,*ptrC,*ptrDen,*ptrMin,*ptrMax;
	double tmpDen;
	const int *DimPatterns;
	register long i;
	register int NdxComp,k;
	double *ptrData;
	
	#if (DEBUG_MODE == 1) 
		/* Log variables */
		FILE * fich;
		char *fileName;
		mwSize loglen; 
		mxArray *Log;
		
		/* Get the name of the log file */
		Log = mxGetField(prhs[0],0,"Log");
		loglen = mxGetNumberOfElements(Log) + 1;
		fileName = mxMalloc(loglen*sizeof(char));

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
	NumFrames=DimPatterns[3];
	ptrData = (double *)mxGetData(prhs[1]);
	NumPixels = NumImageRows * NumImageColumns;

	#if (DEBUG_MODE == 1) 
		fprintf(fich,"Building of the output model\n");
    #endif
	
	/* Duplicate the model structure */
    plhs[0]=mxDuplicateArray(prhs[0]);
        
	#if (DEBUG_MODE == 1) 
		fprintf(fich,"Getting the work variables\n");
	#endif
	
	/* Get the work variables */
	ptrNumCompGauss=mxGetPr(mxGetField(plhs[0],0,"NumCompGauss"));
	ptrCurrentFrame=mxGetPr(mxGetField(plhs[0],0,"CurrentFrame")); 
	ptrNumCompUnif=mxGetPr(mxGetField(plhs[0],0,"NumCompUnif"));
	ptrNumComp=mxGetPr(mxGetField(plhs[0],0,"NumComp"));
	ptrPi=mxGetPr(mxGetField(plhs[0],0,"Pi"));	
	
	Mu=mxGetField(plhs[0],0,"Mu");
	MuFore=mxGetField(plhs[0],0,"MuFore");
	C=mxGetField(plhs[0],0,"C");
    Min=mxGetField(plhs[0],0,"Min");
    Max=mxGetField(plhs[0],0,"Max");
	Den=mxGetField(plhs[0],0,"Den");
	
	NumCompGauss =(int)(*ptrNumCompGauss);
	NumCompUnif = (int)(*ptrNumCompUnif);
	NumComp=(int)(*ptrNumComp);
	
	#if (DEBUG_MODE == 1) 
		fprintf(fich,"Allocating space for work variables\n");
	#endif
	
	/* Allocate space for work variables */
	data = (double*) mxMalloc(NumFrames * Dimension * sizeof(double)); 
    
	/* Work pointers */
	ptrMu = mxGetPr(Mu);
	ptrMuFore = mxGetPr(MuFore);
	ptrC = mxGetPr(C);
	ptrDen = mxGetPr(Den);
	ptrMax = mxGetPr(Max);
	ptrMin = mxGetPr(Min);

	/* Frame counter is initialised */
	*ptrCurrentFrame = 0;

	#if (DEBUG_MODE == 1) 
		fprintf(fich,"Image Width:%d Image Height:%d Dim. of the color space:%d \n", NumImageRows,NumImageColumns,Dimension);
		fprintf(fich,"Gaussian distributions: %d Uniform distributions: %d In total: %d\n",NumCompGauss,NumCompUnif,NumComp);
	#endif

	/* Prepare to compute maxima and minima of the features */
	for (k=0;k<Dimension;k++) 
	{
		ptrMin[k]=DBL_MAX;
		ptrMax[k]=-DBL_MAX;
	}	

	/* For each one of the image pixels */
	for (i=0;i<NumPixels;i++)
	{
		/* The features of this pixel are obtained along the sequence */
		GetPositionData(ptrData+i,data,Dimension,NumPixels,NumFrames);

		#if (DEBUG_MODE == 1)
			if (i==MI_PIXEL) {
				fprintf(fich,"Initialising Pixel nº %d\n",i);
				fprintf(fich,"Data \n");
				RecordMatrixLog(fich,data,Dimension,NumFrames);
			}
		#endif
		
		/* Gaussians distributions are initialised */
		for (NdxComp=0;NdxComp<NumCompGauss;NdxComp++) 
		{
			/* The a priori probability is equally intialised for all the distributions: gaussians and uniforms  */
			ptrPi[NdxComp] = (1.0/(double)NumComp);
			
            /* Both mean and covariance matrix are obtained for each pixel. A tiny value is added to C to avoid 
			singularities */
			MyMean(ptrMu,data,Dimension,NumFrames);
			MyCov(ptrC,data,Dimension,NumFrames);  
			SumDiagonalConstant(ptrC,1.0e-20,ptrC,Dimension);
		
			/* Some information of the process is saved in the log file */
			#if (DEBUG_MODE == 1)
				if (i==MI_PIXEL) {
					fprintf(fich,"Pi: %f\n",*(ptrPi+NdxComp));
					fprintf(fich,"Mu\n");
					RecordMatrixLog(fich,ptrMu,1,Dimension); 
					fprintf(fich,"C\n");
					RecordMatrixLog(fich,ptrC,Dimension,Dimension); 
				}
			#endif
			
			/* Pointers are incremented */
            ptrMu+=Dimension;
            ptrC+=Dimension*Dimension;			
		}

		/* Uniform distributions are initialised */
		for (k=0;k<Dimension;k++) 
		{
			if (data[k]<ptrMin[k])
			{
				ptrMin[k]=data[k];
			}
			if (data[k]>ptrMax[k])
			{
				ptrMax[k]=data[k];
			}
		}
		for (NdxComp=0;NdxComp<NumCompUnif;NdxComp++) 
		{	
			ptrPi[NdxComp+NumCompGauss] = (1.0/(double)NumComp);
		}

		ptrPi+=NumComp;
	}

	/* Compute the density for the uniform distributions */
	tmpDen=1.0;
	for (k=0;k<Dimension;k++) 
	{
		tmpDen *= (ptrMax[k] - ptrMin[k]);
	}
	*ptrDen = 1.0/tmpDen;

	/* Initialize the MuFore vectors */
	for (i=0;i<NumPixels;i++)
	{
		for (NdxComp=0;NdxComp<NumCompGauss;NdxComp++) 
		{
			for (k=0;k<Dimension;k++) 
			{
				ptrMuFore[k] = 0.5*(ptrMax[k] + ptrMin[k]);
			}
			ptrMuFore+=Dimension;
		}
	}

	#if (DEBUG_MODE == 1)
		fprintf(fich,"End of the initialisation process\n");
		fprintf(fich,"-------------------------------------------\n");
		/* Close the log file */
		CerrarLog(fich);
	#endif

	
	/* Release dynamic memory and instances */
	mxFree(data);
}

/* Procedure to return the color intensity of an image pixel which is pointed by 'data'. 
    The number of available frames is indicated in 'NumFrames' */
void GetPositionData(double * data, double * output, int Dimension, int NumPixels, int NumFrames)
{
	double * pDataCurrent=data;
	double * ptr;
	long FrameSize;
	register int i,j;
	
	/* The image size is computed */
	FrameSize = NumPixels*Dimension;
	ptr = output;

	/* The pixel is stored by getting its value in each frame */
	for (i=0;i<NumFrames;i++) 
	{
		for (j=0;j<Dimension;j++) 
		{
			*ptr = *(pDataCurrent+j*NumPixels);
			ptr++;
		}
		pDataCurrent = pDataCurrent+FrameSize;
	}
}

/* Procedure to compute the mean of a pixel distribution  */
void MyMean(double *ptrMean,double *data,int Dimension,int NumPatterns)
{
     int NdxPattern;
     
     memset(ptrMean,0,Dimension*sizeof(double));
     for(NdxPattern=0;NdxPattern<NumPatterns;NdxPattern++)
     {
          MatrixSum(ptrMean,data+NdxPattern*Dimension,ptrMean,Dimension,1);
     }
     ScalarMatrixProduct(1.0/NumPatterns,ptrMean,ptrMean,Dimension,1);
}

/* Procedure to compute the covariance matrix of a pixel distribution  */
void MyCov(double *ptrCov,double *data,int Dimension,int NumPatterns)
{
     double *ptrMean;
     double *ptrDif,*ptrDifDif;
     int NdxPattern;
     
     ptrMean=(double *) mxMalloc(Dimension*sizeof(double));
     ptrDif=(double *) mxMalloc(Dimension*sizeof(double));     
     ptrDifDif=(double *) mxMalloc(Dimension*Dimension*sizeof(double));     
     memset(ptrCov,0,Dimension*sizeof(double));     
     MyMean(ptrMean,data,Dimension,NumPatterns);
     for(NdxPattern=0;NdxPattern<NumPatterns;NdxPattern++)
     {
          MatrixDifference(data+NdxPattern*Dimension,ptrMean,ptrDif,Dimension,1);
          MatrixProduct(ptrDif,ptrDif,ptrDifDif,Dimension,1,Dimension);
          MatrixSum(ptrCov,ptrDifDif,ptrCov,Dimension,Dimension);          
     }
     ScalarMatrixProduct(1.0/NumPatterns,ptrCov,ptrCov,Dimension,Dimension);
     
     mxFree(ptrMean);
     mxFree(ptrDif);
     mxFree(ptrDifDif);          
}

