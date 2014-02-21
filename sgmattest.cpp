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
        data(0,0)=1.0*cos(M_PI/3.0);
        data(0,1)=2.0*cos(M_PI/3.0);
        data(0,2)=3.0*cos(M_PI/3.0);
        data(1,0)=1.0*sin(M_PI/3.0);
        data(1,1)=2.0*sin(M_PI/3.0);
        data(1,2)=3.0*sin(M_PI/3.0);


	float64_t data1[4] = {2.,3.,4.,5.};
	SG_FREE(data.matrix);	
	data.matrix = data1;
	data.num_rows = 2;
	data.num_cols = 2;
	data.display_matrix(data.matrix, 2,2,"data");
        exit_shogun();

        return 0;
}

