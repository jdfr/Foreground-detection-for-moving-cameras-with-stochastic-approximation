#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include <iostream>

#include "opencv2/core/core.hpp"

#include "tbb/tbb.h"
//#include "tbb/tbbmalloc_proxy.h"

using namespace tbb;

#ifndef HAS_OPENCV
#define HAS_OPENCV
#endif

//#include "mex.h"
#include "BMArgs.h"

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


void aplica_transformacion6_MEX(Args_aplica_transformacion6_MEX *args) {
    
	//if (nrhs!=13)
        //printf("aplica_transformacion:invalidArgs: Wrong number of arguments\n");

    //Leemos parametros de entrada de Matlab
	
	int tam_cols = args->tam_cols;
	int tam_rows = args->tam_rows;
	int tam_objgrid = args->tam_objgrid;
	int tam_cols_ini = args->tam_cols_ini;
	int tam_rows_ini = args->tam_rows_ini;
	int nchannels = args->nchannels;

	cv::Mat objgridX = cv::Mat(tam_objgrid, 1, CV_32FC1, args->arg0);
	cv::Mat objgridY = cv::Mat(tam_objgrid, 1, CV_32FC1, args->arg1);

	cv::Mat Mu = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels), args->arg2);
	cv::Mat MuFore = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels), args->arg3);
	cv::Mat R = cv::Mat(tam_rows_ini, tam_cols_ini, CV_MAKETYPE(CV_64F, nchannels*nchannels), args->arg4);
	cv::Mat Pi = cv::Mat(tam_rows_ini, tam_cols_ini, CV_64FC2, args->arg5);
	cv::Mat Counter = cv::Mat(tam_rows_ini, tam_cols_ini, CV_64FC1, args->arg6);
  
	if( Mu.empty() || MuFore.empty() || R.empty() || Pi.empty() || Counter.empty()){ 
		printf("aplica_transformacion:invalidArgs: Error reading Matrix\n");
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
	
	cv::Mat Counter2, Pi2;

	copyMakeBorder(Corona, Corona, tam_rows, tam_rows, tam_cols, tam_cols, cv::BORDER_CONSTANT, 1);
	cv::Mat Corona_out(tam_objgrid, 1, CV_32FC1, args->oarg5);
	cv::remap(Corona, Corona_out, objgridX, objgridY,  cv::INTER_LINEAR, cv::BORDER_CONSTANT, 1);

	cv::Mat Pi_out(tam_objgrid, 1, CV_64FC2,args->oarg4);
	copyMakeBorder(Pi, Pi2, tam_cols, tam_cols, tam_rows, tam_rows, cv::BORDER_REPLICATE);
	cv::remap(Pi2, Pi_out, objgridY, objgridX, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

	cv::Mat Counter_out(tam_objgrid, 1, CV_64FC1,args->oarg2);
	copyMakeBorder(Counter, Counter2, tam_cols, tam_cols, tam_rows, tam_rows, cv::BORDER_REPLICATE);	
	cv::remap(Counter2, Counter_out, objgridY, objgridX,  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
	

	//Return output images to mxArray (Matlab matrix)
	cv::Mat Mu_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels), args->oarg0);
	cv::merge(Mu_nD, Mu_out);

	cv::Mat MuFore_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels), args->oarg1);
	cv::merge(MuFore_nD, MuFore_out);

	cv::Mat R_out(tam_objgrid, 1, CV_MAKETYPE(CV_64F, nchannels*nchannels), args->oarg3);
	cv::merge(R_nD, R_out);

	delete Datos;
}

