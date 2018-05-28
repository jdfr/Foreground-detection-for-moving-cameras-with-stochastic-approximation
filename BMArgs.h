#ifndef _BMArgs_H

#define _BMArgs_H

typedef struct {
  //input arguments
  const char *logFileName;
  double *secondArgData;
  int secondArgDims[4];
  //these are used both for in/out args, as they were taken from a plhs var created like this: plhs[0]=mxDuplicateArray(prhs[0]);
  int *NumCompGauss, *CurrentFrame, *NumComp, *NumCompUnif;
  double *Pi, *Mu, *MuFore, *C, *Min, *Max, *Den;
} Args_initializeBM_MEX;

void initializeBM_MEX(Args_initializeBM_MEX *args);

typedef struct {
  //output was created like this:
  //	mwSize dims_2[2];
  //	dims_2[0]=3;
  //	dims_2[1]=3;
  //	plhs[0]=mxCreateNumericArray (2, dims_2, mxDOUBLE_CLASS, mxREAL);
  double *arg0, *arg1, *output;
  double ransacReproj;
  int minHessian, tam_cols, tam_rows, tam_cols_ext, tam_rows_ext;
  bool upright; //always true??
} Args_extrae_transformacion_BF2_MEX;

void extrae_transformacion_BF2_MEX(Args_extrae_transformacion_BF2_MEX *args);

typedef struct {
  double *arg0, *arg1, *arg2, *arg3, *arg4, *arg5, *arg6;
  int tam_cols, tam_rows, tam_objgrid, tam_cols_ini, tam_rows_ini, nchannels;
	//dims_2[0]=tam_objgrid;
	//dims_2[1]=1;
	//plhs[5]=mxCreateNumericArray (2, dims_2, mxSINGLE_CLASS, mxREAL);
  float *oarg5;
	//dims_3[0]=2;
	//dims_3[1]=tam_objgrid;
	//dims_3[2]=1;
	//plhs[4]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
  double *oarg4;  
	//dims_2[0]=tam_objgrid;
	//dims_2[1]=1;
	//plhs[2]=mxCreateNumericArray (2, dims_2, mxDOUBLE_CLASS, mxREAL);
  double *oarg2;
	//dims_3[0]=nchannels;
	//dims_3[1]=tam_objgrid;
	//dims_3[2]=1;
	//plhs[0]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
  double *oarg0;
	//dims_3[0]=nchannels;
	//dims_3[1]=tam_objgrid;
	//dims_3[2]=1;
	//plhs[1]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
  double *oarg1;
	//dims_3[0]=nchannels*nchannels;
	//dims_3[1]=tam_objgrid;
	//dims_3[2]=1;
	//plhs[3]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
  double *oarg3;
} Args_aplica_transformacion6_MEX;

void aplica_transformacion6_MEX(Args_aplica_transformacion6_MEX *args);

typedef struct {
  bool wantOutput;
  const char *logFileName;
  int NumImageRows, NumImageColumns, Dimension; //dims 0, 1, 2 de prhs[1]
  double *arg1;
  //these are used both for in/out args, as they were taken from a plhs var created like this: plhs[0]=mxDuplicateArray(prhs[0]);
  int *NumCompGauss, *CurrentFrame, *NumComp, *NumCompUnif, *Z;
  double *Epsilon, *Pi, *Mu, *MuFore, *C, *LogDetC, *Min, *Max, *Den, *Counter, *Noise;
  //          DimPatterns=mxGetDimensions(prhs[1]);
  //          NumImageRows = DimPatterns[0];
  //          NumImageColumns = DimPatterns[1];
  //          Dimension = DimPatterns[2];
  //          plhs[1] = mxCreateNumericArray(2,DimPatterns,mxDOUBLE_CLASS, mxREAL);
  double *oarg1;
  //          DimResp[0]=NumImageRows;
  //          DimResp[1]=NumImageColumns;
  //          DimResp[2] = NumComp;
  //          plhs[2] = mxCreateNumericArray(3,DimResp,mxDOUBLE_CLASS, mxREAL);
  //double *oarg2; //THIS SEEMS UNUSED
} Args_updateBM_MEX;

void updateBM_MEX(Args_updateBM_MEX *args);

#endif

