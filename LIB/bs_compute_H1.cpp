#include <math.h>
#include <mex.h>
#include <omp.h>

/*
 * For windows:
 * mex bs_compute_H1.cpp COMPFLAGS="/openmp $COMPFLAGS"
 * For Linux:
 * v1: mex bs_compute_H1.cpp CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
 * v2: mex bs_compute_H1.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
 */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(!mxIsSingle(prhs[0])||mxIsComplex(prhs[0])||mxGetNumberOfDimensions(prhs[0])!=2)
    {
        mexErrMsgTxt("The first input must be single, real, two dimensional matrix..."); //A
    }
    if(!mxIsInt32(prhs[1])||mxIsComplex(prhs[1])) //x
    {
        mexErrMsgTxt("The second input must be int32, real, one column vector......");
    }
    if(!mxIsInt32(prhs[2])||mxIsComplex(prhs[2]))//y
    {
        mexErrMsgTxt("The third input must be int32, real, one column vector...");
    }
    if(!mxIsSingle(prhs[3])||mxIsComplex(prhs[3])) //v -- non-zeros of W
    {
        mexErrMsgTxt("The fourth input must be single, real, two dimensional matrix...");
    }
    
    float *A = (float*) mxGetPr(prhs[0]);
    int *x = (int*) mxGetPr(prhs[1]);
    int *y = (int*) mxGetPr(prhs[2]);
    float *v = (float*) mxGetPr(prhs[3]);
    int N = mxGetM(prhs[0]);
    int non_zero_num = mxGetM(prhs[1]);
    
    plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float *J = (float*) mxGetPr(plhs[0]);
    float tmp_J = 0;
    
#pragma omp parallel for shared(A, x, y, v, N) reduction(+:tmp_J) schedule(static)
        
        for(int iter_a = 0; iter_a < non_zero_num; iter_a++)
        {
            //mexPrintf("Num threads %d.\n", omp_get_thread_num());
            //num_threads(100)
            for(int iter_b = 0; iter_b < non_zero_num; iter_b++)
            { 
                float tmp1 = A[x[iter_b] + x[iter_a]*N];
                float tmp2 = A[y[iter_b] + y[iter_a]*N];
                tmp_J += v[iter_a]*v[iter_b]*(tmp1-tmp2)*(tmp1-tmp2);
            }
        }
        J[0] = tmp_J;
}

