#include "Debugging.h"
#include "mex.h"
#include <math.h>




void PrintValues(double *Values,int NumValues)
{
    int ndx;
    
    for(ndx=0;ndx<NumValues;ndx++)
        mexPrintf("%lf\n",Values[ndx]);
    mexPrintf("\n\n");
}
  
    
void PrintMatrix(double *Matrix,int NumRows,int NumCols)
{
    int NdxRow,NdxCol;
    
    for(NdxRow=0;NdxRow<NumRows;NdxRow++)
    {
        for(NdxCol=0;NdxCol<NumCols;NdxCol++)
        {
            mexPrintf("%lf\t",Matrix[NdxCol*NumRows+NdxRow]);
        }
        mexPrintf("\n");
    }        
    mexPrintf("\n\n");
}  




