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

void XXh(double* out, double* xf, double* T, double* Uh, double* UUh, double h, int N){
    double * tmp = malloc(sizeof(double)*N*N);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                tmp[i*N+j] += UUh[i*N+k]*T[j*N+k];
            }
        }
    }
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                out[i*N+j] += T[i*N+k]*tmp[k*N+j];
            }
        }
    }
    free(tmp);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            out[i*N+j] += h*xf[i]*xf[j];
            for (int k=0;k<N;k++){
                out[i*N+j] += xf[i]*T[j*N+k]*Uh[k] + xf[j]*T[i*N+k]*Uh[k];
            }
        }
    }
}
