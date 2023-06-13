#include <stdlib.h>
#include <math.h>

static int isclose(double a, double b){
    return fabs(a-b) <= 1e-8 + 1e-5*fabs(b);
}

static double fn(double si, double sj, double h){
    if (isclose(si,-sj)){
        return h;
    }else{
        return (exp((si + sj)*h) - 1.) / (si + sj);
    }
}
void UUh(double* out, double* u0, double* eigs, double h, int N){
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            out[i*N+j] = u0[i] * u0[j] * fn(eigs[i],eigs[j],h);
        }
    }
}

