#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/preprocessor/PCA.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
        fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
        init_shogun(&print_message, &print_message, &print_message);

        SGMatrix<float64_t> data(2,3);
        data(0,0)=1.0;
        data(0,1)=2.0;
        data(0,2)=3.0;
        data(1,0)=1.0;
        data(1,1)=2.0;
        data(1,2)=3.0;
	
//	CDenseFeatures<float64_t>* feat =new  CDenseFeatures<float64_t>(data);
//	SGMatrix<float64_t> m = feat->get_feature_matrix(); 

//	data.matrix = (float64_t*) realloc(m.matrix,4*sizeof(float64_t));
	float64_t* data1 = NULL;
	data1 = data.matrix;
        data.matrix = (float64_t*) realloc(data1,4*sizeof(float64_t));

//	SG_FREE(data.matrix);	
//	data.matrix = data1;
	data.num_rows = 2;
	data.num_cols = 2;
        SG_SPRINT("data(0,0) : %f", data(0,0));
//        SG_SPRINT("data1[0] : %f", data1[0]);
	data.display_matrix(data.matrix, 2,2,"data");
//        SG_UNREF(feat);

	exit_shogun();

        return 0;
}

