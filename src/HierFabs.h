#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <Rmath.h>
#include <stdio.h>
#include <stdlib.h>
#define QEPS 1e-6

// Loss function
double Loss(double *y, double *xb, int model, int *param, int *stautus, double tau);

double calculate_bic(double *score, int *param, int model, double gamma);

// Heap sort for an double array, from big to small 
void swap(double *arr, int *set,int i,int j);

void heapify(double *arr, int *set, int i, int n);

void HeapSortDouble(double *arr, int *set, int n);

// backward qualify for weak hierarchy, judge if a main effect k can vanish 
int b_qlf(int k, int *param, int *active, int *parent, int *child);

void Pre_work(double *x, double *y, double *weight, int *param, 
    double *meanx, double *sdx, int isq);

// update sparse index
void usi(int *s_i, int *s_j, int *tt_b, int *tt_a, int *act, int ns, int iter);

// unmormalized_beta
void unmormalized(double *beta, double *unbeta, int *parent, int *indexi, 
    int *indexj, int iter, double *intercept, double *meanx, double *sdx, 
    double meany, int total_active, int model);

void der(double *x, double *y, double *d, double *meanx, double *sdx, int model, 
    int *param, double *residual, double *xb, int isq, int *stautus, int *active,
    int hierarchy, double tau);


// take an initial step
void Initial(double *x, double *y, double *xb, double *beta, double *d, double *w, 
    int *param, int model, double eps, double *lambda, double *losses, int *direction, 
    int *active, double *bic, int *df, double *meanx, double *sdx, double *residual, 
    int isq, int *stautus, int hierarchy, double gamma, double tau);

// try a backward step
int Backward(double *x, double *y, double *xb, double *beta, double *d, double *betanew, 
    double *weight, int *param, int model, double eps, double *lambda, 
    double *losses, int *parent, double xi, int back, int hierarchy, int *active, int *direction, double *bic, int *df, 
    int *child, double *meanx, double *sdx, double *residual, int isq, int *stautus, double gamma, double tau);

// used in forward, to find the forward coordinate
void compare(double *temp, double value, int *k, int h12, int *k2, int t, 
    int *parenti, int i, int *parentj, int j);

// take an forward step
void Forward(double *x, double *y, double *xb, double *beta, double *d, double *weight, int *param, int model, double eps, double *lambda, double *losses, 
    int *parent, double xi, int back, int hierarchy, int *direction, int *active, double *bic, int *df, int *child, double *meanx, 
    double *sdx, double *residual, int isq, int *stautus, double gamma, double tau);


int HFabs(double *x, double *y, double *xb, double *beta, double *d, double *weight, int *param, int model, double eps, double *lambda, double *losses, int *parent, 
    double xi, int back, int hierarchy, int *direction, int *active, double *bic, int *df, int *child, double *meanx, double *sdx, double *residual, 
    int isq, int *stautus, int *sparse_i, int *sparse_j, int *iter, int stoping, double lam_m, int max_s, double gamma, double tau);


void Pre_work_GE(double *x, double *z, double *y, double *weight, int *param, 
    double *meanx, double *sdx);


// backward qualify for weak hierarchy, judge if a main effect k can vanish 
int b_qlf_GE(int k, int *param, int *active, int *parent, int *child);


void der_GE(double *x, double *z, double *y, double *d, double *meanx, double *sdx, int model, 
    int *param, double *residual, double *xb, int *stautus, int hierarchy, int *active, double tau);

// take an initial step
void Initial_GE(double *x, double *z, double *y, double *xb, double *beta, double *d, double *w, 
    int *param, int model, double eps, double *lambda, double *losses, int *direction, 
    int *active, double *bic, int *df, double *meanx, double *sdx, double *residual, 
    int *stautus, int hierarchy, double gamma, double tau);

// try a backward step
int Backward_GE(double *x, double *z, double *y, double *xb, double *beta, double *d, double *betanew, 
    double *weight, int *param, int model, double eps, double *lambda, 
    double *losses, int *parent, double xi, int back, int hierarchy, int *active, int *direction, double *bic, int *df, 
    int *child, double *meanx, double *sdx, double *residual, int *stautus, double gamma, double tau);


// take an forward step
void Forward_GE(double *x, double *z, double *y, double *xb, double *beta, double *d, double *weight, 
    int *param, int model, double eps, double *lambda, double *losses, 
    int *parent, double xi, int back, int hierarchy, int *direction, int *active, double *bic, int *df, int *child, double *meanx, 
    double *sdx, double *residual, int *stautus, double gamma, double tau);


int HFabs_GE(double *x, double *z, double *y, double *xb, double *beta, double *d, double *weight, 
    int *param, int model, double eps, double *lambda, double *losses, int *parent, 
    double xi, int back, int hierarchy, int *direction, int *active, double *bic, 
    int *df, int *child, double *meanx, double *sdx, double *residual, 
    int *stautus, int *sparse_i, int *sparse_j, int *iter, int stoping, double lam_m, int max_s, double gamma, double tau);


SEXP Hierarchy_Fabs(SEXP X, SEXP Z, SEXP Y, SEXP Weight, SEXP Model, SEXP Epsilon, SEXP Lam_min, 
    SEXP Xi, SEXP Back, SEXP Stoping, SEXP Iter, SEXP Hierarchy, SEXP Param, 
    SEXP Max_S, SEXP MeanY, SEXP Isquadratic, SEXP Status, SEXP GE, SEXP Gamma, SEXP TAU);
