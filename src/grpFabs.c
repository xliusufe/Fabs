#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <Rmath.h>
#include <stdio.h>
#include <stdlib.h>

#include "grpFabs.h"


/* spr
this is a interface function for compute the  parital rank correlation and
the smoothed (using sigmoid function ) partial rank correlation.*/
double spr(double *y, double *x, double *b, int *status, double *sigma, int *param)
{
    int i, j; 
	int n = param[0], p = param[1]; 
    double c[n], u, corr;
    
    for (i = 0; i < n; ++i){
    	c[i] = 0;
    	for (j = 0; j < p; ++j)
    		c[i] += x[i*p + j]* b[j];
    }

    corr = 0.0;
    for (i=0; i<n; i++){
         for (j=0; j<n; j++){
         	if (j != i){
                u = (c[i]-c[j])/sigma[0];
		        corr += 1.0*status[j] * (y[i] > y[j]) / (1 + exp(-1.0*u));	 
      		}
		}
    }

    return corr;
}


void dspr(y, x, b, status, sigma, param, derivative)
double *y, *x, *b, *sigma, *derivative;
int *status, *param;
{
	int i, j, k;
	int n = param[0], p = param[1]; 
    double  a0, c[n], u;
    
    // derivative doesn't need initial value
    for (i = 0; i < p; ++i) derivative[i] = 0.0;

    for (i = 0; i < n; ++i){
    	c[i] = 0;
    	for (j = 0; j < p; ++j)
    		c[i] += x[i*p + j]* b[j];
    }

    for (i=0; i<n; i++){
        for (j=0; j<n; j++){ 
            if (j != i){
                u =  (c[i] - c[j])/sigma[0];

                if (u < -500) continue; // a0 = 0;
                else a0 = exp(-1.0*u)/pow(1.0 + exp(-1.0*u), 2.0)/sigma[0];

                for (k=0; k<p; k++)
                    derivative[k] += 1.0*status[j] * (y[i] > y[j]) * a0 * (x[i*p+k] - x[j*p+k]);
            }
        }
    }   	
}

/*coxloss
the cox loss function.*/
double coxloss(double *y, double *x, double *b, int *status, int *param)
{
	int i, j; 
	int n = param[0], p = param[1]; 
    double c[n], loss = 0.0;
    
    for (i = 0; i < n; ++i){
    	c[i] = 0;
    	for (j = 0; j < p; ++j)
    		c[i] += x[i*p + j]* b[j];
    }

    double temp = 0.0;
    for (i = n-1; i >= 0; --i){
        temp += exp(c[i]);
        if (status[i]) loss -= c[i] - log(temp);
    }

    return loss;
}

void dcox(y, x, b, status, param, derivative)
double *y, *x, *b, *derivative;
int *status, *param;
{
	int i, j, k;
	int n = param[0], p = param[1]; 
    double  xb[n], temp, theta = 0.0;
    
    // derivative doesn't need initial value
    for (i = 0; i < p; ++i) derivative[i] = 0.0;

    for (i = 0; i < n; ++i){
    	xb[i] = 0;
    	for (j = 0; j < p; ++j)
    		xb[i] += x[i*p + j]* b[j];
    }

    for (i = n-1; i >= 0; --i) {
        theta += exp(xb[i]);
        if (status[i]) {
            for (j = i; j < n; ++j) {
                temp = exp(xb[j])/theta;
                for (k = 0; k < p; ++k) {
                    derivative[k] -= temp * (x[i*p+k]-x[j*p+k]);
                }
            }
        }
    }
}

/*logisticloss
the logistic loss function.*/
double logisticloss(double *y, double *x, double *b, int *param)
{
	int i, j; 
	int n = param[0], p = param[1]; 
    double c[n], loss;
    
    for (i = 0; i < n; ++i){
    	c[i] = 0;
    	for (j = 0; j < p; ++j)
    		c[i] += x[i*p + j]* b[j];
    }

    loss = 0.0;
    for (i = 0; i < n; ++i) loss += log(1+exp(-y[i]*c[i]));

    return loss;
}

void dlogstic(y, x, b, param, derivative)
double *y, *x, *b, *derivative;
int *param;
{
	int i, j;
	int n = param[0], p = param[1]; 
    double  c[n], u;
    
    // derivative doesn't need initial value
    for (i = 0; i < p; ++i) derivative[i] = 0.0;

    for (i = 0; i < n; ++i){
    	c[i] = 0;
    	for (j = 0; j < p; ++j)
    		c[i] += x[i*p + j]* b[j];
    }

    for (i=0; i<n; i++){
    	u = c[i] * y[i];
    	if (u > 500) continue;
    	else {
    		for (j = 0; j < p; ++j)
    			derivative[j] -= y[i] * x[i*p+j] / (exp(u)+1.0);
    	}
    }   	
}

// LossGFabs function
double LossGFabs(y, X, b, status, model, sigma, param)
double *y, *X, *b, *sigma;
int *status, model, *param;
{
	int i,j;
	int n = param[0], p = param[1];
	double temp, val;

	val = 0.0;
    if(model == 1) {
    	for (i = 0; i < n; ++i){
    		temp = 0.0;
    		for (j = 0; j < p; ++j)
    			temp += X[i*p+j] * b[j];
    		val += pow(y[i] - temp, 2)/2.0/n;
    	}
    } else if(model == 2) {
    	val = -1.0*spr(y, X, b, status, sigma, param)/n/(n-1);
    } else if (model == 3) {
    	val = coxloss(y, X, b, status, param);
    } else if (model == 4) {
    	val = logisticloss(y, X, b, param);
    }

    return val;
}

double Score(y, X, b, status, model, sigma, param)
double *y, *X, *b, *sigma;
int *status, model, *param;
{
	int i,j;
	int n = param[0], p = param[1];
	double temp, val;

	val = 0.0;
    if(model == 1) {
    	for (i = 0; i < n; ++i){
    		temp = 0.0;
    		for (j = 0; j < p; ++j)
    			temp += X[i*p+j] * b[j];
    		val += pow(y[i] - temp, 2)/2.0/n;
    	}
    } else if(model == 2) {
    	val = spr(y, X, b, status, sigma, param)/n/(n-1);
    } else if (model == 3) {
    	val = coxloss(y, X, b, status, param);
    } else if (model == 4) {
    	val = logisticloss(y, X, b, param);
    }

    return val;
}


double mynorm(double *x, int l0, int l1, int *type)
{
	double norm = 0.0;

	if (type[0] == 2)
		for (int i = l0; i <= l1; ++i) 
			norm += x[i] * x[i];
	
	return sqrt(norm);
}


double crossproduct(double *x, double *y, int l0, int l1)
{
	double norm = 0.0;

	for (int i = l0; i <= l1; ++i) norm += x[i] * y[i];

	return norm;
}

/*
a function to calculate derivatives and find optimal index k in the active set.
for spr estimator, no anchor is selected.*/
int derivativeInitial(x, y, b0, w, status, k0, k1, model, sigma, type, param, d)
double *x, *y, *b0, *w, *sigma, *d;
int *status, *k0, *k1, model, *type, *param;
{
    // d doesn't need initial value
    int i, j, k = -1;
    int n = param[0], p = param[1], pg = param[2];
    double d_gr, temp;

    if(model == 1){
    	double residual[n];
        for (i = 0; i < n; ++i){
            residual[i] = y[i];
            for (j = 0; j < p; ++j)
                residual[i] -= x[i*p+j] * b0[j];
        }

        for (i = 0; i < p; ++i){
            d[i] = 0.0;
            for (j = 0; j < n; ++j)
                d[i] += x[i+j*p] * residual[j];
            d[i] /= (-2.0*n);
        }
    }else if(model == 2){
        dspr(y, x, b0, status, sigma, param, d);
        for (i = 0; i < p; ++i)
            d[i] /= (-1.0*n*(n-1));
    } else if (model == 3) {
    	dcox(y, x, b0, status, param, d);
    } else if (model == 4) {
    	dlogstic(y, x, b0, param, d);
    }

    temp = 0.0;
    for (i = 0; i < pg; ++i){
        d_gr = mynorm(d, k0[i], k1[i], type)/w[i];
        if (d_gr > temp){
            temp = d_gr;
            k = i;
        }
    }

    return k;
}


int derivativeBack(x, y, b, w, status, k0, k1, model, sigma, type, Aset, param, d)
double *x, *y, *b, *w, *sigma, *d;
int *status, *k0, *k1, model, *type, *Aset, *param;
{
    // d doesn't need initial value
    int i, j, k;
    int n = param[0], p = param[1], ns = param[3], set[ns];
    double residual[n], d_gr, temp;
    for (i = 0; i < ns; ++i) set[i] = Aset[i]-1;

    if(model == 1){
        for (i = 0; i < n; ++i){
            residual[i] = y[i];
            for (j = 0; j < p; ++j)
                residual[i] -= x[i*p+j] * b[j];
        }

        for (i = 0; i < p; ++i){
            d[i] = 0.0;
            for (j = 0; j < n; ++j)
                d[i] += x[i+j*p] * residual[j];
            d[i] /= (-2.0*n);
        }
    }else if(model == 2){
        dspr(y, x, b, status, sigma, param, d);
        for (i = 0; i < p; ++i)
            d[i] /= (-1.0*n*(n-1));
    } else if (model == 3) {
    	dcox(y, x, b, status, param, d);
    } else if (model == 4) {
    	dlogstic(y, x, b, param, d);
    }

    temp = crossproduct(d, b, k0[set[0]], k1[set[0]])/mynorm(b, k0[set[0]], k1[set[0]], type)/w[set[0]];
    k = set[0];
    for (i = 1; i < ns; ++i){
        d_gr = crossproduct(d, b, k0[set[i]], k1[set[i]])/mynorm(b, k0[set[i]], k1[set[i]], type)/w[set[i]];
        if (d_gr > temp){
            temp = d_gr;
            k = set[i];
        }
    }

    return k;
}

// bic for varying coefficient model
double bic(x, y, beta, status, set, sigma, sum_status, param, model, vc)
double *x, *y, *beta, *sigma;
int *status, *set, *sum_status, *param, model, vc;
{
	double score;
	int i, j, n, ns, count;

	n = param[0];
	ns = param[3];
	count = ns;
	score = Score(y, x, beta, status, model, sigma, param);
	
	if (vc)
	{
		for (i = 0; i < ns; ++i) {
			if (set[i] % 2 == 1) {
				for (j = 0; j < ns; ++j)
					if (set[j] == set[i] + 1) count -= 1;
			}
		}
	}

	if(model == 1) {
		return -log(score) - count * log(n)/2/n;
    } else if(model == 2) {
    	return log(score) - count * log(sum_status[0])/2/n;
    } else if (model == 3) {
    	return log(score) - count * log(sum_status[0])/2/n;
    } else if (model == 4) {
    	return -log(score) - count * log(n)/2/n;
    } else {
    	return -log(score) - count * log(n)/2/n;
    }
}

int argmax(double *array, int length)
{
	double maximal = array[0];
	int index = 0;

	for (int i = 1; i < length; ++i) {
		if (array[i] > maximal) {
			maximal = array[i];
			index = i;
		}
	}

	return index;
}


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
		SEXP Stoping, SEXP Iter, SEXP Param, SEXP VC)
{
	int i, j, l, k, p;
	double *x, *y, *w;
	double *sigma, eps, lam_m, xi;
	int *status, *k0, *k1, *param;
	int model, *type, back, stoping, iter, vc;
	const char *modeltype = CHAR(asChar(Model));
	if (strcmp(modeltype, "spr") == 0)
	{
		model = 2;
	} else if (strcmp(modeltype, "square") == 0)
	{
		model = 1;
	} else if (strcmp(modeltype, "cox") == 0)
	{
		model = 3;
	} else if (strcmp(modeltype, "logistic") == 0)
	{
		model = 4;
	} else {
		model = 1;
	}

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
	vc      = INTEGER(VC)[0];
	type    = INTEGER(Type);
	back    = INTEGER(Back)[0];
	stoping = INTEGER(Stoping)[0];
	iter    = INTEGER(Iter)[0];
	p       = param[1];

	SEXP Beta, Lambda, Direction, Active, Aset[iter], Loops, Result, R_names, BIC, Optimal;
	double *b0, *d, lossa, lossb, norma, normb, descent;
	double *beta, *lambda, *BI;
	int *direction, sum_status[1];

	b0 = (double*)malloc(sizeof(double)*p);
	d = (double*)malloc(sizeof(double)*p);
	beta = (double*)malloc(sizeof(double)*iter*p);
	lambda = (double*)malloc(sizeof(double)*iter);
	direction = (int*)malloc(sizeof(int)*iter);
	BI = (double*)malloc(sizeof(double)*iter);

	sum_status[0] = 0;
	for (j = 0; j < param[0]; ++j) sum_status[0] += status[j];

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
	BI[0] = bic(x, y, beta, status, INTEGER(Aset[0]), sigma, sum_status, param, model, vc);
	
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

		BI[i+1] = bic(x, y, beta+(i+1)*p, status, INTEGER(Aset[i+1]), sigma, sum_status, param, model, vc);
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
	PROTECT(BIC       = allocVector(REALSXP, i));
	PROTECT(Loops     = allocVector(INTSXP, 1));
	PROTECT(Optimal   = allocVector(INTSXP, 1));
	PROTECT(Result    = allocVector(VECSXP, 7));
	PROTECT(R_names   = allocVector(STRSXP, 7));

	INTEGER(Loops)[0] = i;
	INTEGER(Optimal)[0] = argmax(BI, i) + 1;
	for (j = 0; j < i*p; ++j) REAL(Beta)[j] = beta[j];
	for (j = 0; j < i; ++j) {
		REAL(BIC)[j] = BI[j];
		REAL(Lambda)[j] = lambda[j];
		INTEGER(Direction)[j] = direction[j];
		SET_VECTOR_ELT(Active, j, Aset[j]);
	}

	free(b0);
	free(d);
	free(beta);
	free(lambda);
	free(direction);
	free(BI);
	
	char *names[7] = {"beta", "lambda", "direction", "active", "iter", "bic", "opt"};
	for(j = 0; j < 7; ++j)
		SET_STRING_ELT(R_names, j,  mkChar(names[j]));
	SET_VECTOR_ELT(Result, 0, Beta);
	SET_VECTOR_ELT(Result, 1, Lambda);
	SET_VECTOR_ELT(Result, 2, Direction);
	SET_VECTOR_ELT(Result, 3, Active);
	SET_VECTOR_ELT(Result, 4, Loops); 
	SET_VECTOR_ELT(Result, 5, BIC);
	SET_VECTOR_ELT(Result, 6, Optimal);    
	setAttrib(Result, R_NamesSymbol, R_names); 

	UNPROTECT(9+i);
	return Result;
}





