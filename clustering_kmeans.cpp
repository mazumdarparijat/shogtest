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

void minibatchKMeans(int32_t k, int32_t b, int32_t t, CDistance* d)
{
	CDenseFeatures* lhs=(CDenseFeatures*) d->get_lhs();
	int32_t XSize=lhs->get_num_vectors();
	int32_t dims=lhs->get_num_features();

	SGMatrix<float64_t> C=SGMatrix<float64_t>(dims,k);
	SGVector<float64_t> v=SGVector<float64_t>(k);
	v.zero();

	SGVector<int32_t> crand=choose_rand(k,XSize);
	for (int32_t i=0; i<k; i++)
	{
		SGVector<float64_t> feature=lhs->get_feature_vector(i);
		for (int32_t j=0; j<dims; j++)
			C(j,i)=feature[j];
	}

	CDenseFeatures<float64_t>* rhs_mus = new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache = d->replace_rhs(rhs_mus);
	rhs_mus->set_feature_matrix(C);


	for (int32_t i=0; i<t; i++)
	{
		SGVector<int32_t> M=choose_rand(b,XSize);
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
		
		for (int32_t j=0; j<b; j++)
		{
			int32_t near=ncent[j]
			SGVector<float64_t> c_alive=rhs_mus->get_feature_vector(near);
			SGVector<float64_t> x=lhs->get_feature_vector(M[j]); 
			v[near]+=1.0;
			float64_t eta=1.0/v[near];
			c_alive=(1-eta)*c_alive + eta*x;
		}
	}

}

SGVector<int32_t> choose_rand(int32_t b, int32_t num)
{
	SGVector<int32_t> chosen=SGVector<int32_t>(num);
	SGVector<int32_t> ret=SGVector<int32_t>(b);
	chosen.zero();
	int32_t ch=0;
	while (ch<b)
	{
		const int32_t n=CMath::random(0,num);
		if (chosen[n]==0)
		{
			chosen[n]+=1;
			ret[ch]=n;
			ch++;
		}
	}
	return ret;
}

void set_random_centers(float64_t* weights_set, int32_t* ClList, int32_t XSize, CDistance* distance, SGMatrix<float64_t> mus)
{
        CDenseFeatures<float64_t>* lhs=
                        (CDenseFeatures<float64_t>*)distance->get_lhs();

        for (int32_t i=0; i<XSize; i++)
        {
                const int32_t Cl=CMath::random(0, k-1);
                weights_set[Cl]+=1.0;
                ClList[i]=Cl;

                int32_t vlen=0;
                bool vfree=false;
                float64_t* vec=lhs->get_feature_vector(i, vlen, vfree);

                for (int32_t j=0; j<dimensions; j++)
                        mus.matrix[Cl*dimensions+j] += vec[j];

                lhs->free_feature_vector(vec, i, vfree);
        }

        SG_UNREF(lhs);

        for (int32_t i=0; i<k; i++)
        {
                if (weights_set[i]!=0.0)
                {
                        for (int32_t j=0; j<dimensions; j++)
                                mus.matrix[i*dimensions+j] /= weights_set[i];
                }
        }
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
	CKMeans* clustering=new CKMeans(2, distance);

	for (int32_t i=0; i<5; i++)
	{
		clustering->train(features);

		CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());
	
//		for (index_t i=0; i<result->get_num_labels(); ++i)
//			SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));
		
		SG_SPRINT("kmeans w/o kmeans++ result %i \n",i+1);

		CDenseFeatures<float64_t>* centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
		SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();
		SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows, centers_matrix.num_cols, "learnt centers");
	

		SG_UNREF(centers);
		SG_UNREF(result);
	}

	cout<<"done"<<endl;

	clustering->set_use_kmeanspp(true);
	for (int32_t i=0; i<5; i++)
	{
		clustering->train(features);

		CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());
	
//		for (index_t i=0; i<result->get_num_labels(); ++i)
//			SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));
		
		SG_SPRINT("kmeans with kmeans++ result %i \n",i+1);

		CDenseFeatures<float64_t>* centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
		SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();
		SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows, centers_matrix.num_cols, "learnt centers");
	

		SG_UNREF(centers);
		SG_UNREF(result);
	}

	SG_UNREF(clustering);
//	SG_UNREF(clusteringpp);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}

