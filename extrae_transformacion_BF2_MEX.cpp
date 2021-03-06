#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifndef HAS_OPENCV
#define HAS_OPENCV
#endif

//#include "tbb/tbbmalloc_proxy.h"

#include "BMArgs.h"
#include <stdio.h>

//Usando BruteForce matcher
void extrae_transformacion_BF2_MEX(Args_extrae_transformacion_BF2_MEX *args) {
    
	//if (nrhs!=9)
        //printf("extrae_tranformacion:invalidArgs: Wrong number of arguments\n");

	int tam_cols = args->tam_cols;
	int tam_rows = args->tam_rows;

	int tam_cols_ext = args->tam_cols_ext;
	int tam_rows_ext = args->tam_rows_ext;

    //Read Matlab image and load it to an Mat struct
	cv::Mat img_ant = cv::Mat(tam_rows, tam_cols, CV_8UC3, args->arg0).t();
	cv::Mat img_act = cv::Mat(tam_rows_ext, tam_cols_ext, CV_8UC3, args->arg1).t();
	bool upright = args->upright;
	int minHessian = args->minHessian;
	double ransacReproj = args->ransacReproj;
   
	if( img_ant.empty() || img_act.empty() ){ 
		printf("extrae_tranformacion:invalidArgs: Error reading images\n");
	}

	//-- Step 1: Detect the keypoints and descriptors (feautre vectors) using SURF Detector
	cv::Mat descriptors_ant, descriptors_act;
	std::vector<cv::KeyPoint> keypoints_ant, keypoints_act;

	cv::SURF detector( minHessian, 4, 2, true, upright );

	detector(img_ant, cv::noArray(), keypoints_ant, descriptors_ant);
	detector(img_act, cv::noArray(), keypoints_act, descriptors_act);

	// Step 2: Matching descriptor vectors using FLANN matcher
	cv::BFMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors_ant, descriptors_act, matches );
	
	double max_dist = 0; double min_dist = 100;

	// Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_ant.rows; i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}


	// Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	//Cuanto mayor sea el umbral, mas matches se consideraran buenos
	std::vector< cv::DMatch > good_matches;
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	int idxMatches = 0;
	for( int i = 0; i < descriptors_ant.rows; i++ ){ 
		if( matches[i].distance < 9*min_dist ){ 
			good_matches.push_back( matches[i]);

			// Get the keypoints from the good matches
			obj.push_back( keypoints_ant[ good_matches[idxMatches].queryIdx ].pt );
			scene.push_back( keypoints_act[ good_matches[idxMatches].trainIdx ].pt );

			idxMatches++;
		}
	}

	// Localize the object
	cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC, ransacReproj);
   
    //Return output image to mxArray (Matlab matrix)
	cv::Mat H_out(3, 3, CV_64FC1,args->output);
	H=H.t();
	H.copyTo(H_out);
}
