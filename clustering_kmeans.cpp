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
using namespace shogun;
using namespace std;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

SGVector<int32_t> choose_rand(int32_t b, int32_t num);

void minibatchKMeans(int32_t k, int32_t b, int32_t t, CDistance* d)
{
	CDenseFeatures<float64_t>* lhs=(CDenseFeatures<float64_t>*) d->get_lhs();
	int32_t XSize=lhs->get_num_vectors();
	cout<<"XSize : "<<XSize<<endl;
	int32_t dims=lhs->get_num_features();
	cout<<"dims : "<<dims<<endl;
	SGMatrix<float64_t> C=SGMatrix<float64_t>(dims,k);
	SGVector<float64_t> v=SGVector<float64_t>(k);
	v.zero();

	SGVector<int32_t> crand=choose_rand(k,XSize);
	crand.display_vector("crand");
	for (int32_t i=0; i<k; i++)
	{
		SGVector<float64_t> feature=lhs->get_feature_vector(crand[i]);
		for (int32_t j=0; j<dims; j++)
			C(j,i)=feature[j];
	}

	CDenseFeatures<float64_t>* rhs_mus = new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache = d->replace_rhs(rhs_mus);
	rhs_mus->set_feature_matrix(C);
	for (int32_t i=0; i<t; i++)
	{
	        SGMatrix<float64_t>::display_matrix(C.matrix,dims,k,"fast kmeans");	
		cout<<"i : "<<i<<endl;
		SGVector<int32_t> M=choose_rand(b,XSize);
		M.display_vector("M");
		SGVector<int32_t> ncent= SGVector<int32_t>(b);
		for (int32_t j=0; j<b; j++)
		{
			
			SGVector<float64_t> dists=SGVector<float64_t>(k);
			for (int32_t p=0; p<k; p++)
				dists[p]=d->distance(M[j],p);

			int32_t imin=0;
			float64_t min=dists[0];
			for (int32_t p=1; p<k; p++)
			{
				if (dists[p]<min)
				{
					imin=p;
					min=dists[p];
				}
			}
			ncent[j]=imin;
		}
		ncent.display_vector("ncent");
		for (int32_t j=0; j<b; j++)
		{
			int32_t near=ncent[j];
			SGVector<float64_t> c_alive=rhs_mus->get_feature_vector(near);
			c_alive.display_vector("c");
			SGVector<float64_t> x=lhs->get_feature_vector(M[j]);
			x.display_vector("x");
			v[near]+=1.0;
			float64_t eta=1.0/v[near];
			cout<<"eta : "<<eta<<endl;
			c_alive.scale((1-eta));
			c_alive.display_vector("c scaled");
			x.scale(eta);
			x.display_vector("x scaled");
			c_alive=c_alive + x;
			c_alive.display_vector("c final");
			rhs_mus->set_feature_vector(c_alive, near);
		}
	}
	SGMatrix<float64_t>::display_matrix(C.matrix,dims,k,"fast kmeans");
}

SGVector<int32_t> choose_rand(int32_t b, int32_t num)
{
	SGVector<int32_t> chosen=SGVector<int32_t>(num);
	SGVector<int32_t> ret=SGVector<int32_t>(b);
	chosen.zero();
	int32_t ch=0;
	while (ch<b)
	{
		const int32_t n=CMath::random(0,num-1);
		cout<<"SEE : "<<n<<endl;
		if (chosen[n]==0)
		{
			chosen[n]+=1;
			ret[ch]=n;
			ch++;
		}
	}
	return ret;
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

	minibatchKMeans(1,4,100,distance);
//	SG_UNREF(clustering);
//	SG_UNREF(clusteringpp);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}

