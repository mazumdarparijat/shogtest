/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Parijat Mazumdar
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/Math.h>
#include <iostream>
#include "mbKMeans.h"
using namespace shogun;
using namespace std;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	int32_t dim_features=2;
	
	/* create data around clusters */
	SGMatrix<float64_t> data(dim_features, 4);
	data(0,0) = 0;
	data(0,1) = 0;
	data(0,2) = 2;
	data(0,3) = 2;
	data(1,0) = 0;
	data(1,1) = 1000;
	data(1,2) = 1000;
	data(1,3) = 0;
	SGMatrix<float64_t>::display_matrix(data.matrix, 2,
			4, "rectangle_coordinates");



	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t> (data);
	SG_REF(features);

	CEuclideanDistance* distance = new CEuclideanDistance(features, features);
//	CKMeans* clustering=new CKMeans(2, distance);


//	clustering->train(features);

//	CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());
	
//		for (index_t i=0; i<result->get_num_labels(); ++i)
//			SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));

//	CDenseFeatures<float64_t>* centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
//	SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();
//	SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows, centers_matrix.num_cols, "learnt centers");
	

//	SG_UNREF(centers);
//	SG_UNREF(result);

	cout<<"done"<<endl;

	mbKMeans(2,2,100,distance);
//	SG_UNREF(clustering);
//	SG_UNREF(clusteringpp);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}

