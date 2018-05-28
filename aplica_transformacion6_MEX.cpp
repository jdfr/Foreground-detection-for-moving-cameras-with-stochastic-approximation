#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include <iostream>

#include "opencv2/core/core.hpp"

#include "tbb/tbb.h"
#include "tbb/tbbmalloc_proxy.h"

using namespace tbb;

#ifndef HAS_OPENCV
#define HAS_OPENCV
#endif

#include "mex.h"

//mex -g aplica_transformacion6_MEX.cpp opencv_imgproc248.lib opencv_core248.lib -IC:\opencv\build\include

struct AlmacenDatos {
       std::vector<cv::Mat> *Mu_nD, *MuFore_nD, *R_nD;
	   int nchannels, tam_rows, tam_rows_end, tam_cols, tam_cols_end;
	   cv::Mat objgridX, objgridY;
        
       AlmacenDatos(std::vector<cv::Mat> *Mu, std::vector<cv::Mat> *MuFore, 
		   std::vector<cv::Mat> *R, int num_channels, int t_r, int t_r_end, int t_c, int t_c_end,
		   cv::Mat objgridX_in, cv::Mat objgridY_in){

            Mu_nD = Mu;
			MuFore_nD = MuFore;
			R_nD = R;

			nchannels =num_channels;
			tam_rows= t_r;
			tam_rows_end = t_r_end;
			tam_cols = t_c;
			tam_cols_end = t_c_end;
			
			objgridX = objgridX_in;
			objgridY = objgridY_in;
        }
};

struct AplicadorParalelo {
	AlmacenDatos *const Datos;

	void copyMakeBorder_nD(std::vector<cv::Mat> *src, int tam_rows, int tam_rows_end, int tam_cols, int tam_cols_end, 
						   int chInincial, int chFinal) const{
		for(int i=chInincial;i<=chFinal;++i){
			copyMakeBorder((*src)[i], (*src)[i], tam_rows, tam_rows_end, tam_cols, tam_cols_end, cv::BORDER_REPLICATE);
		}
	}

	void remap_N_canales(std::vector<cv::Mat> *src, cv::Mat objgridX, cv::Mat objgridY, int interp, int border, 
							int valor, int chInincial, int chFinal) const{
		for(int i=chInincial;i<=chFinal;++i){
			cv::remap((*src)[i], (*src)[i], objgridX, objgridY, interp, border, valor);
		}
	}
	void operator()( const blocked_range<size_t>& r ) const {
		int nchannels, tam_rows, tam_rows_end, tam_cols, tam_cols_end;

		nchannels = Datos->nchannels;
		tam_rows=Datos->tam_rows;
		tam_rows_end=Datos->tam_rows_end;
		tam_cols=Datos->tam_cols;
		tam_cols_end=Datos->tam_cols_end;

		//Extendemos las vbles del modelo para cubrir los desplazamiento de la imagen actual
		copyMakeBorder_nD(Datos->Mu_nD, tam_cols, tam_cols_end, tam_rows, tam_rows_end, r.begin(), r.end()-1);
		copyMakeBorder_nD(Datos->MuFore_nD, tam_cols, tam_cols_end, tam_rows, tam_rows_end, r.begin(), r.end()-1);
		copyMakeBorder_nD(Datos->R_nD, tam_cols, tam_cols_end, tam_rows, tam_rows_end, r.begin()*nchannels, ((r.end()-1)*nchannels+nchannels-1));

		remap_N_canales(Datos->Mu_nD, Datos->objgridY, Datos->objgridX,  cv::INTER_LINEAR, 
			cv::BORDER_REPLICATE, 0, r.begin(), r.end()-1);
		remap_N_canales(Datos->MuFore_nD, Datos->objgridY, Datos->objgridX,  cv::INTER_LINEAR, 
			cv::BORDER_REPLICATE, 0, r.begin(), r.end()-1);
		remap_N_canales(Datos->R_nD, Datos->objgridY, Datos->objgridX,  cv::INTER_LINEAR, 
			cv::BORDER_REPLICATE,0, r.begin()*nchannels, ((r.end()-1)*nchannels+nchannels-1));
	}

	AplicadorParalelo(AlmacenDatos *const aux_Datos) : Datos(aux_Datos)
    {
    }
        
	~AplicadorParalelo(){
	}
};

cv::Mat transponer_N_canales(cv::Mat src, int nchannels){
	cv::Mat res;
	std::vector<cv::Mat> channelsSrc(nchannels);

	cv::split(src, channelsSrc);
	for(int i=0;i<nchannels;i++){
		channelsSrc[i]=channelsSrc[i].t();
	}

	cv::merge(channelsSrc, res);

	return res;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
        const mxArray *prhs[]) {
    
	if (nrhs!=13)
        mexErrMsgIdAndTxt("aplica_transformacion:invalidArgs", "Wrong number of arguments");

    //Leemos parametros de entrada de Matlab
	
	int tam_cols = (int) *mxGetPr(prhs[7]);
	int tam_rows = (int) *mxGetPr(prhs[8]);
	int tam_objgrid = (int) *mxGetPr(prhs[9]);
	int tam_cols_ini = (int) *mxGetPr(prhs[10]);
	int tam_rows_ini = (int) *mxGetPr(prhs[11]);
	int nchannels = (int) *mxGetPr(prhs[12]);

	cv::Mat objgridX = cv::Mat(tam_objgrid, 1, CV_32FC1, mxGetPr(prhs[0]));
	cv::Mat objgridY = cv::Mat(tam_objgrid, 1, CV_32FC1, mxGetPr(prhs[1]));

	cv::Mat Mu = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels), mxGetPr(prhs[2]));
	cv::Mat MuFore = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels), mxGetPr(prhs[3]));
	cv::Mat R = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels*nchannels), mxGetPr(prhs[4]));
	cv::Mat Pi = cv::Mat(tam_rows_ini, tam_cols_ini, CV_64FC2, mxGetPr(prhs[5]));
	cv::Mat Counter = cv::Mat(tam_rows_ini, tam_cols_ini, CV_64FC1, mxGetPr(prhs[6]));
  
	if( Mu.empty() || MuFore.empty() || R.empty() || Pi.empty() || Counter.empty()){ 
		mexErrMsgIdAndTxt("aplica_transformacion:invalidArgs", "Error reading Matrix");
	}

	cv::Mat Corona;
	
	Corona = cv::Mat::ones(Mu.cols, Mu.rows, CV_32FC1);
	Corona = Corona*0.5;

	std::vector<cv::Mat> Mu_nD(nchannels), MuFore_nD(nchannels), R_nD(nchannels*nchannels);
	cv::split(Mu, Mu_nD);
	cv::split(MuFore, MuFore_nD);
	cv::split(R, R_nD);	

	AlmacenDatos *const Datos = new AlmacenDatos(&Mu_nD, &MuFore_nD, &R_nD, nchannels, 
		tam_rows, tam_rows, tam_cols, tam_cols, objgridX, objgridY);
	parallel_for( blocked_range<size_t>(0,nchannels), AplicadorParalelo(Datos));
	
	mwSize dims_2[2], dims_3[3];
	cv::Mat Counter2, Pi2;

	copyMakeBorder(Corona, Corona, tam_rows, tam_rows, tam_cols, tam_cols, cv::BORDER_CONSTANT, 1);
	dims_2[0]=tam_objgrid;
	dims_2[1]=1;
	plhs[5]=mxCreateNumericArray (2, dims_2, mxSINGLE_CLASS, mxREAL);
	cv::Mat Corona_out(tam_objgrid, 1, CV_32FC1, mxGetPr(plhs[5]));
	cv::remap(Corona, Corona_out, objgridX, objgridY,  cv::INTER_LINEAR, cv::BORDER_CONSTANT, 1);

	dims_3[0]=2;
	dims_3[1]=tam_objgrid;
	dims_3[2]=1;
	plhs[4]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
	cv::Mat Pi_out(tam_objgrid, 1, CV_64FC2,mxGetPr(plhs[4]));
	copyMakeBorder(Pi, Pi2, tam_cols, tam_cols, tam_rows, tam_rows, cv::BORDER_REPLICATE);
	cv::remap(Pi2, Pi_out, objgridY, objgridX, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

	dims_2[0]=tam_objgrid;
	dims_2[1]=1;
	plhs[2]=mxCreateNumericArray (2, dims_2, mxDOUBLE_CLASS, mxREAL);
	cv::Mat Counter_out(tam_objgrid, 1, CV_64FC1,mxGetPr(plhs[2]));
	copyMakeBorder(Counter, Counter2, tam_cols, tam_cols, tam_rows, tam_rows, cv::BORDER_REPLICATE);	
	cv::remap(Counter2, Counter_out, objgridY, objgridX,  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
	

	//Return output images to mxArray (Matlab matrix)
	dims_3[0]=nchannels;
	dims_3[1]=tam_objgrid;
	dims_3[2]=1;
	plhs[0]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
	cv::Mat Mu_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels), mxGetPr(plhs[0]));
	cv::merge(Mu_nD, Mu_out);

	dims_3[0]=nchannels;
	dims_3[1]=tam_objgrid;
	dims_3[2]=1;
	plhs[1]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
	cv::Mat MuFore_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels), mxGetPr(plhs[1]));
	cv::merge(MuFore_nD, MuFore_out);

	dims_3[0]=nchannels*nchannels;
	dims_3[1]=tam_objgrid;
	dims_3[2]=1;
	plhs[3]=mxCreateNumericArray (3, dims_3, mxDOUBLE_CLASS, mxREAL);
	cv::Mat R_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels*nchannels), mxGetPr(plhs[3]));
	cv::merge(R_nD, R_out);

	delete Datos;
}

