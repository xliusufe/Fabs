#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <Rmath.h>
#include <stdio.h>
#include <stdlib.h>



double spr(double *y, double *x, double *b, int *status, double *sigma, int *param);


void dspr(double *y, double *x, double *b, int *status, double *sigma, int *param, double *derivative);

/*coxloss
the cox loss function.*/
double coxloss(double *y, double *x, double *b, int *status, int *param);

void dcox(double *y, double *x, double *b, int *status, int *param, double *derivative);

/*logisticloss
the logistic loss function.*/
double logisticloss(double *y, double *x, double *b, int *param);

void dlogstic(double *y, double *x, double *b, int *param, double *derivative);

// LossGFabs function
double LossGFabs(double *y, double *X, double *b, int *status, int model, double *sigma, int *param);

double Score(double *y, double *X, double *b, int *status, int model, double *sigma, int *param);

double mynorm(double *x, int l0, int l1, int *type);


double crossproduct(double *x, double *y, int l0, int l1);

/*
a function to calculate derivatives and find optimal index k in the active set.
for spr estimator, no anchor is selected.*/
int derivativeInitial(double *x, double *y, double *b0, double *w, int *status, int *k0, int *k1, int model, double *sigma, int *type, int *param, double *d);


int derivativeBack(double *x, double *y, double *b, double *w, int *status, int *k0, int *k1, int model, double *sigma, int *type, int *Aset, int *param, double *d);

// bic for varying coefficient model
double bic(double *x, double *y, double *beta, int *status, int *set, double *sigma, int *sum_status, int *param, int model, int vc);

int argmax(double *array, int length);


/*SEXP grpFabs(SEXP X, SEXP Y, SEXP Weight, SEXP Status, SEXP K0, SEXP K1, SEXP Model, 
		SEXP Sigma, SEXP Epsilon, SEXP Lambda_min, SEXP Xi, SEXP Type, SEXP Back, 
		SEXP Stoping, SEXP Iter, SEXP Param)
{
	int i, j, l, k, p;
	double *x, *y, *w;
	double *sigma, eps, lam_m, xi;
	int *status, *k0, *k1, *param;
	int *model, *type, back, stoping, iter;

	x      = REAL(X);
	y      = REAL(Y);
	w      = REAL(Weight);
	status = INTEGER(Status);
	k0     = INTEGER(K0);
	k1     = INTEGER(K1);
	param  = INTEGER(Param);


	sigma   = REAL(Sigma);
	eps     = REAL(Epsilon)[0];
	lam_m   = REAL(Lambda_min)[0];
	xi      = REAL(Xi)[0];
	model   = INTEGER(Model);
	type    = INTEGER(Type);
	back    = INTEGER(Back)[0];
	stoping = INTEGER(Stoping)[0];
	iter    = INTEGER(Iter)[0];
	p       = param[1];

	SEXP Beta, Lambda, Direction, Active, Aset[iter], Loops, Result, R_names;
	double *b0, *d, lossa, lossb, norma, normb, descent;
	double *beta, *lambda;
	int *direction;

	b0 = (double*)malloc(sizeof(double)*p);
	d = (double*)malloc(sizeof(double)*p);
	beta = (double*)malloc(sizeof(double)*iter*p);
	lambda = (double*)malloc(sizeof(double)*iter);
	direction = (int*)malloc(sizeof(int)*iter);


	// step 1: initial step (forward)
	for (i = 0; i < p; ++i) b0[i] = 0.0;
	for (i = 0; i < iter*p; ++i) beta[i] = 0.0;
	
	k = derivativeInitial(x, y, b0, w, status, k0, k1, model, sigma, type, param, d);

	normb = mynorm(d, k0[k], k1[k], type);
	for (i = k0[k]; i <= k1[k]; ++i)
		beta[i] = -1.0*d[i] * eps / normb / w[k];

	lossb        = LossGFabs(y, x, b0, status, model, sigma, param);
	lossa        = LossGFabs(y, x, beta, status, model, sigma, param);
	lambda[0]    = (lossb - lossa)/eps;
	direction[0] = 1;
	param[3]     = 1;
	PROTECT(Aset[0] = allocVector(INTSXP, param[3]));
	INTEGER(Aset[0])[0] = k+1;
	
	// step 2: forward and backward
	for (i = 0; i < iter-1; ++i)
	{
		k = derivativeBack(x, y, beta+i*p, w, status, k0, k1, model, sigma, type, INTEGER(Aset[i]), param, d);

		normb = mynorm(beta+i*p, k0[k], k1[k], type);
		for (j = 0; j < p; ++j) b0[j] = beta[i*p+j];
		for (j = k0[k]; j <= k1[k]; ++j)
			b0[j] -= b0[j] * eps / normb / w[k];

		lossb   = LossGFabs(y, x, beta+i*p, status, model, sigma, param);
		lossa   = LossGFabs(y, x, b0, status, model, sigma, param);
		norma   = mynorm(b0, k0[k], k1[k], type);
		descent = lossa - lossb + lambda[i]*w[k]*(norma - normb);

		if ((descent < -1.0*xi) & (back == 1))
		{
			for (j = 0; j < p; ++j) beta[i*p+p+j] = b0[j];
			lambda[i+1] = lambda[i];
			direction[i+1] = -1;

		} else {
			k = derivativeInitial(x, y, beta+i*p, w, status, k0, k1, model, sigma, type, param, d);
	
			normb  = mynorm(d, k0[k], k1[k], type);
			for (j = 0; j < p; ++j) beta[i*p+p+j] = beta[i*p+j];
			for (j = k0[k]; j <= k1[k]; ++j){
				beta[i*p+p+j] -= d[j] * eps / normb / w[k];
			}

			lossb   = LossGFabs(y, x, beta+i*p, status, model, sigma, param);
			lossa   = LossGFabs(y, x, beta+i*p+p, status, model, sigma, param);
			normb   = mynorm(beta+i*p, k0[k], k1[k], type);
			norma   = mynorm(beta+i*p+p, k0[k], k1[k], type);
			descent = (lossb - lossa - xi) / fabs(w[k] * (norma - normb));

			lambda[i+1] = (lambda[i]<descent) ? lambda[i] : descent;
			direction[i+1] = 1;
		}

		if (norma < xi){
			// setdiff
			for (j = 0; j < param[3]; ++j){
				if (INTEGER(Aset[i])[j] == k+1){
					param[3] -= 1;
					break;
				}
			}
			PROTECT(Aset[i+1] = allocVector(INTSXP, param[3]));
			for (l = 0; l < j; ++l) INTEGER(Aset[i+1])[l] = INTEGER(Aset[i])[l];
			for (l = j; l < param[3]; ++l) INTEGER(Aset[i+1])[l] = INTEGER(Aset[i])[l+1];

			for (j = k0[k]; j <= k1[k]; ++j) beta[i*p+p+j] = 0.0;
		} else if(direction[i+1] == 1) {
			// setunion
			for (j = 0, l = 1; j < param[3]; ++j){
				if (INTEGER(Aset[i])[j] == k+1){
					l = 0;
					break;
				}
			}
			PROTECT(Aset[i+1] = allocVector(INTSXP, param[3]+l));
			for (j = 0; j < param[3]; ++j) INTEGER(Aset[i+1])[j] = INTEGER(Aset[i])[j];
			if (l == 1) INTEGER(Aset[i+1])[param[3]] = k+1;
			param[3] += l;
		} else {
			PROTECT(Aset[i+1] = allocVector(INTSXP, param[3]));
			for (l = 0; l < param[3]; ++l) INTEGER(Aset[i+1])[l] = INTEGER(Aset[i])[l];
		}

		if ( (stoping == 1) & (lambda[i+1] <= lambda[0] * lam_m) ) {
			i += 1;
			break;
		}
		if (i == iter-2) {
			printf("Solution path unfinished, more iterations is needed.\n"); 
		}
	}
	
	i += 1;

	PROTECT(Beta      = allocVector(REALSXP, i*p));
	PROTECT(Lambda    = allocVector(REALSXP, i));
	PROTECT(Direction = allocVector(INTSXP, i));
	PROTECT(Active    = allocVector(VECSXP, i));
	PROTECT(Loops     = allocVector(INTSXP, 1));
	PROTECT(Result    = allocVector(VECSXP, 5));
	PROTECT(R_names   = allocVector(STRSXP, 5));

	INTEGER(Loops)[0] = i;
	for (j = 0; j < i*p; ++j) REAL(Beta)[j] = beta[j];
	for (j = 0; j < i; ++j) {
		REAL(Lambda)[j] = lambda[j];
		INTEGER(Direction)[j] = direction[j];
		SET_VECTOR_ELT(Active, j, Aset[j]);
	}

	free(b0);
	free(d);
	free(beta);
	free(lambda);
	free(direction);
	
	char *names[5] = {"beta", "lambda", "direction", "active", "iter"};
	for(j = 0; j < 5; ++j)
		SET_STRING_ELT(R_names, j,  mkChar(names[j]));
	SET_VECTOR_ELT(Result, 0, Beta);
	SET_VECTOR_ELT(Result, 1, Lambda);
	SET_VECTOR_ELT(Result, 2, Direction);
	SET_VECTOR_ELT(Result, 3, Active);
	SET_VECTOR_ELT(Result, 4, Loops);   
	setAttrib(Result, R_NamesSymbol, R_names); 

	UNPROTECT(7+i);
	return Result;
}*/


SEXP BIC_grpFabs(SEXP X, SEXP Y, SEXP Weight, SEXP Status, SEXP K0, SEXP K1, SEXP Model, 
		SEXP Sigma, SEXP Epsilon, SEXP Lambda_min, SEXP Xi, SEXP Type, SEXP Back, 
		SEXP Stoping, SEXP Iter, SEXP Param, SEXP VC);




