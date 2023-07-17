#include <stdlib.h>
#include <math.h>

struct EccentricityResonanceInteraction {
    int order;
    double* C2_mtrx;
    double* C1_vec;
};

struct EccentricityResonanceInteraction* create_interaction(int indexIn, int indexOut, int kres){
    struct EccentricityResonanceInteraction* interaction = calloc(1,sizeof(struct EccentricityResonanceInteraction));
    interaction->order = (kres%2==0)?1:2;
    interaction->C2_mtrx = calloc(2*2,sizeof(double));
    interaction->C1_vec = calloc(2,sizeof(double));
}

void free_interaction(struct EccentricityResonanceInteraction* interaction){
    if (interaction){
        free(interaction->C2_mtrx);
        free(interaction->C1_vec);
        free(interaction);
    }
}


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

void XXh(double* out, double* xf, double* T, double* Uh, double* u0, double* eigs, double h, int N){
    double * _UUh = malloc(sizeof(double)*N*N);
    UUh(_UUh, u0, eigs, h, N);
    double * tmp = malloc(sizeof(double)*N*N);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                tmp[i*N+j] += _UUh[i*N+k]*T[j*N+k];
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
    free(_UUh);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            out[i*N+j] += h*xf[i]*xf[j];
            for (int k=0;k<N;k++){
                out[i*N+j] += xf[i]*T[j*N+k]*Uh[k] + xf[j]*T[i*N+k]*Uh[k];
            }
        }
    }
}
