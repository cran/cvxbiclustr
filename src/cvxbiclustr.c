#include <R.h> 
#include <Rmath.h> 
#include <math.h>
#include <stdio.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))

typedef struct {
  int Nrow, Ncol;
  double *data;
} matrix;

typedef struct {
  int Nrow, Ncol;
  int *column_ptr;
  int *row_indices;
  double *values;
} spmatrix;

void kernel_weights(double *X, int *p, int *n, double *phi, double *w) {
  int i, j, k, l;
  double sos;
  k = 0;
  for (i=0; i<*n-1; i++)
    for (j=i+1; j<*n; j++) {
      sos = 0.;
      for (l=0; l<*p; l++)
	sos += pow(X[l + (*p)*i]-X[l + (*p)*j],2.);
      w[k] = exp(-(*phi)*sos);
      k += 1;
    }
}

/*
 Sparse Matrix / Dense Matrix multiply : Y = M*X
*/
void spmm(spmatrix M, matrix X, matrix Y) {
  int i, j, k;
  int m, n, p;
  
  m = M.Nrow;
  n = M.Ncol;
  p = X.Ncol;
  
  Y.Nrow = m;
  Y.Ncol = p;
  
  for (i=0; i < m*p; i++)
    Y.data[i] = 0.;
  
  for (i=0; i < p; ++i)
    for (j=0; j < n; ++j)
      for (k=M.column_ptr[j]; k < M.column_ptr[j+1]; ++k)
	Y.data[M.row_indices[k] + m*i] += M.values[k]*X.data[j + n*i];
}

/*
  Sparse Matrix / Dense Matrix multiply : Y = M'*X
*/
void spmtm(spmatrix M, matrix X, matrix Y) {
  int i, j, k;
  int m, n, p;
  
  m = M.Nrow;
  n = M.Ncol;
  p = X.Ncol;
  
  for (i=0; i < n*p; i++)
    Y.data[i] = 0.;
  
  for (i=0; i < p; ++i)
    for (j=0; j < n; ++j)
      for (k=M.column_ptr[j]; k < M.column_ptr[j+1]; ++k)
	Y.data[j + n*i] += M.values[k]*X.data[M.row_indices[k] + m*i];
}

/*
  Sparse Matrix / Dense Matrix multiply : Y = M*X'
*/
void spmmt(spmatrix M, matrix X, matrix Y) {
  int i, j, k;
  int m, n, p;
  
  m = M.Nrow;
  n = M.Ncol;
  p = X.Nrow;
  
  Y.Nrow = m;
  Y.Ncol = p;
  
  for (i=0; i < m*p; i++)
    Y.data[i] = 0.;
  
  for (i=0; i < p; ++i)
    for (j=0; j < n; ++j)
      for (k=M.column_ptr[j]; k < M.column_ptr[j+1]; ++k)
	Y.data[M.row_indices[k] + m*i] += M.values[k]*X.data[i + p*j];
}

/*
  Objective function for Lagrangian dual of convex clustering problem.
  dual(Lambda) = -0.5||X-Lambda*Phi||_F^2 + 0.5||X||_F^2
               = -0.5||U||_F^2 + 0.5||X||_F^2
*/
void convex_cluster_dual(matrix XT, matrix UT, double *output) {
  int i;
  int n, p;
  double dual = 0.;
  
  n = XT.Nrow;
  p = XT.Ncol;
  
  for (i=0; i<n*p; i++)
    dual += XT.data[i]*XT.data[i] - UT.data[i]*UT.data[i];
  dual *= 0.5;
  *output = dual;
}

/*
  Objective function of convex clustering problem.
  primal(U) = 0.5||X - U||_F^2 + J(U)
*/
void convex_cluster_primal(matrix XT, matrix UT, matrix VT, spmatrix Phi, double *w, double *output) {
  int i, j;
  int m, n, p;
  double norm_row;
  double primal = 0.;
  double penalty = 0.;

  m = Phi.Nrow;
  n = XT.Nrow;
  p = XT.Ncol;

  for (i=0; i<n*p; i++)
    primal += pow(XT.data[i] - UT.data[i],2.);
  primal = 0.5*primal;
  
  // V^t = Phi * U^t
  spmm(Phi, UT, VT);

  // Compute penalty
  for (i=0; i<m; i++) {
    // Compute 2-norm of ith row.
    norm_row = 0.;
    for (j=0; j<p; j++)
      norm_row += pow(VT.data[i + m*j],2.);
    penalty += w[i]*sqrt(norm_row);
  }

  *output = primal + penalty;
}

/*
 Objective function of convex biclustering problem.
 primal(U) = 0.5||X - U||_F^2 + J(U) + J(U^t)
 */
void convex_bicluster_primal(matrix XT, matrix UT, matrix VT_row, matrix VT_col,
			     spmatrix Phi_row, spmatrix Phi_col,
			     double *w_row, double *w_col, double *output) {
    int i, j;
    int m_row, m_col, n, p;
    double norm;
    double primal = 0.;
    double penalty = 0.;
    
    m_row = Phi_row.Nrow;
    m_col = Phi_col.Nrow;
    n = XT.Nrow;
    p = XT.Ncol;
    
    for (i=0; i<n*p; i++)
      primal += pow(XT.data[i] - UT.data[i],2.);
    primal = 0.5*primal;
    
    // V_row^t = Phi_row * U
    spmmt(Phi_row, UT, VT_row);
    
    // V_col^t = Phi_col * U^t
    spmm(Phi_col, UT, VT_col);
    
    // Compute penalty
    for (i=0; i<m_row; i++) {
      // Compute 2-norm of ith row difference
      norm = 0.;
      for (j=0; j<n; j++)
	norm += pow(VT_row.data[i + m_row*j],2.);
      penalty += w_row[i]*sqrt(norm);
    }
    for (i=0; i<m_col; i++) {
      // Compute 2-norm of ith column difference
      norm = 0.;
      for (j=0; j<p; j++)
	norm += pow(VT_col.data[i + m_col*j],2.);
      penalty += w_col[i]*sqrt(norm);
    }
    
    *output = primal + penalty;
}

/*
  Apply proximal mapping of 2-norm to the rows of a matrix.
  X - input matrix
  Y - output matrix
  tau - is an array of proximal mapping parameters - one for each row
*/
void prox_L2(matrix X, matrix Y, double *tau) {
  int i, j;
  int m, n;
  double norm_row;
  
  m = X.Nrow;
  n = X.Ncol;
  
  for (i=0; i<m; i++) {
    // Compute 2-norm of ith row.
    norm_row = 0.;
    for (j=0; j<n; j++)
      norm_row += pow(X.data[i + m*j],2.);
    norm_row = sqrt(norm_row);
    // Case 1: ith row is zero vector.
    if (norm_row == 0.)
      for (j=0; j<n; j++)
	Y.data[i + m*j] = X.data[i + m*j];
    // Case 2: ith row is non the zero vector.
    else
      for (j=0; j<n; j++)
	Y.data[i + m*j] = fmax(0.,1.-(tau[i]/norm_row))*X.data[i + m*j];
  }
}

/*
  Test prox_L2.
*/
void test_prox_L2(double *X, double *Y, double *tau, int *m, int *n) {
  matrix Xm, Ym;
  Xm.Nrow = *m;
  Xm.Ncol = *n;
  Xm.data = X;
 
  Ym.Nrow = *m;
  Ym.Ncol = *n;
  Ym.data = Y;
  
  prox_L2(Xm, Ym, tau);
}

/*
  Euclidean projection of the rows of matrix onto 2-norm balls of radii tau.
  X - input matrix
  Y - output matrix
  tau - is an array of proximal mapping parameters - one for each row
 */
void proj_L2(matrix X, matrix Y, double *tau) {
  int i,j;
  int m,n;
  double norm_row;
  
  m = X.Nrow;
  n = X.Ncol;

  for (i=0; i<m; i++) {
    // Compute 2-norm of ith row.
    norm_row = 0.;  
    for (j=0; j<n; j++)
      norm_row += pow(X.data[i + m*j],2.);
    norm_row = sqrt(norm_row); 
    // Case 1: ith row is outside of the tau radius ball.
    if (norm_row > tau[i])
      for (j=0; j<n; j++)
	Y.data[i + m*j] = (tau[i]/norm_row)*X.data[i + m*j];
    // Case 2: ith row is inside of the tau radius ball.
    else
      for (j=0; j<n; j++)
	Y.data[i + m*j] = X.data[i + m*j];
  }
}

/*
  U^t = X^t - Phi^t * Lambda^t
*/
void update_UT(matrix XT, matrix LambdaT, matrix UT, spmatrix Phi) {
  int i;

  // U^t <- Phi^t * Lambda^t  
  spmtm(Phi, LambdaT, UT);
  // U^t <- X^t - U^t
  for (i=0; i<(UT.Nrow*UT.Ncol); i++)
    UT.data[i] = XT.data[i] - UT.data[i];
}

void grad_LambdaT(matrix UT, spmatrix Phi, matrix gLambdaT) {
  int i;
  
  spmm(Phi,UT,gLambdaT);
  
  for (i=0; i<gLambdaT.Nrow*gLambdaT.Ncol; i++)
    gLambdaT.data[i] = -gLambdaT.data[i];
}

/*
  Lambda^t = proj_L2(Lambda^t + nu*gLambdaT)
 */
void update_LambdaT2(matrix LambdaT, matrix LambdaT_temp, matrix gLambdaT, double *nu, double *w) {
  int i;
  int m, p;
  m = LambdaT.Nrow;
  p = LambdaT.Ncol;
  
  // Lambda^t <- Proj(Lambda^t - nu*gLambdaT)
 
  for (i=0; i<m*p; i++)
    LambdaT_temp.data[i] = LambdaT.data[i] - (*nu)*gLambdaT.data[i];
  proj_L2(LambdaT_temp,LambdaT,w);
  
}

/*
  Lambda^t = proj_L2(Lambda^t + nu*Phi*U^t)
 */
void update_LambdaT(matrix LambdaT, matrix LambdaT_temp, matrix UT, spmatrix Phi, double *nu,
		    double *w) {
  int i;
  int m, n, p;
  m = Phi.Nrow; // LambdaT.Nrow
  n = Phi.Ncol; // UT.Nrow
  p = LambdaT.Ncol; // UT.Ncol 
  
  // Lambda^t_temp <- Phi * UT
  spmm(Phi,UT,LambdaT_temp);
  // Lambda^t <- Proj(Lambda^t + nu*Lambda^t_temp)
 
  for (i=0; i<m*p; i++)
    LambdaT_temp.data[i] = LambdaT.data[i] + (*nu)*LambdaT_temp.data[i];
  proj_L2(LambdaT_temp,LambdaT,w);
  
}

/*
  Update VT_row = prox_tau(Phi_row * U)
*/
void update_VT_row(matrix U, matrix LambdaT, matrix VT, spmatrix Phi,
		   double *w, double *nu) {
  int i, m, n;
  matrix VT_temp;
  double *tau;
  
  m = LambdaT.Nrow;
  n = LambdaT.Ncol;
  
  VT_temp.Nrow = m;
  VT_temp.Ncol = n;
  VT_temp.data = (double*) calloc(m*n,sizeof(double));
  
  tau = (double*) calloc(m,sizeof(double));
  for (i=0; i<m; i++)
    tau[i] = w[i]/(*nu);
  
  // VT_temp = Phi * U
  spmm(Phi, U, VT_temp);
  
  // VT_temp = VT_temp - (1/nu)*LambdaT
  for (i=0; i<m*n; i++)
    VT_temp.data[i] -= (1/(*nu))*LambdaT.data[i];
  
  // VT = prox_L2(VT_temp, tau)
  prox_L2(VT_temp, VT, tau);
  
  free(VT_temp.data);
  free(tau);
}

/*
  Update VT_col = prox_tau(Phi_col * UT)
*/
void update_VT_col(matrix UT, matrix LambdaT, matrix VT, spmatrix Phi,
                   double *w, double *nu) {
  int i, m, n;
  matrix VT_temp;
  double *tau;
  
  m = LambdaT.Nrow;
  n = LambdaT.Ncol;
  
  VT_temp.Nrow = m;
  VT_temp.Ncol = n;
  VT_temp.data = (double*) calloc(m*n,sizeof(double));
    
  tau = (double*) calloc(m,sizeof(double));
  for (i=0; i<m; i++)
    tau[i] = w[i]/(*nu);
  
  // VT_temp = Phi * UT
  spmm(Phi, UT, VT_temp);
  
  // VT_temp = VT_temp - (1/nu)*LambdaT
  for (i=0; i<m*n; i++)
    VT_temp.data[i] -= (1/(*nu))*LambdaT.data[i];
  
  // VT = prox_L2(VT_temp, tau)
  prox_L2(VT_temp, VT, tau);
  
  free(VT_temp.data);
  free(tau);
}

/*
  Convex clustering via AMA
*/
void convex_cluster(matrix XT, matrix UT, 
		    matrix VT, matrix LambdaT, matrix LambdaT_temp,
		    spmatrix Phi, double *w, double *nu,
                    double *primal, double *dual, int *max_iter,
                    int *iter, double *tol) {
    
  int its;
  double fp, fd;

  for (its=0; its<*max_iter; its++) {
    update_UT(XT, LambdaT, UT, Phi);
    update_LambdaT(LambdaT, LambdaT_temp, UT, Phi, nu, w);
    convex_cluster_primal(XT, UT, VT, Phi, w, &fp);
    primal[its] = fp;
    convex_cluster_dual(XT, UT, &fd);
    dual[its] = fd;
    if (fp-fd < *tol) break;
  }
  if (its == *max_iter)
    its -= 1;
  *iter = its;
  
}

/*
 Convex clustering using Barzilai-Borwein adaptive step size.
 */
void convex_cluster_fasta(matrix XT, matrix UT,
			  matrix VT, matrix LambdaT, matrix LambdaT_temp,
			  matrix LambdaT_old, matrix dLambdaT,
			  matrix gLambdaT, matrix gLambdaT_old,
			  spmatrix Phi, double *w, double *nu,
			  double *primal, double *dual,
			  int *max_iter, int *iter, double *tol) {
  int i, its, j;
  int m, n, p;
  int backtrack_count;
  int M = 10;
  double dLambdaSq, dLambdaDotdGrad, dGradSq, dg;
  double nu_m, nu_s;
  double dual_temp, primal_temp;
  double del, dual_local_max, lhs, rhs;
    
  m = Phi.Nrow;
  n = Phi.Ncol;
  p = UT.Ncol;
  
  // Compute initial dual loss
  update_UT(XT, LambdaT, UT, Phi);
  convex_cluster_dual(XT, UT, &dual_temp);
  dual[0] = dual_temp;
  
  // grad_LambdaT = -Phi * U^t
  grad_LambdaT(UT, Phi, gLambdaT);
  
  for (its=1; its<*max_iter; its++) {
    
    // Store last Lambda^t variables
    for (i=0; i<m*p; i++)
      LambdaT_old.data[i] = LambdaT.data[i];
    
    // Forward-Backward Step
    update_LambdaT2(LambdaT, LambdaT_temp, gLambdaT, nu, w);
    
    // Compute the dual objective at proposed iterate
    update_UT(XT, LambdaT, UT, Phi);
    convex_cluster_dual(XT, UT, &dual_temp);
    
    // Non-monotone backtracking line search
    dual_local_max = -dual[its-1];
    for (j=its-2; j>=max(its-M,0); j--) {
      dual_local_max = fmax(dual_local_max,-dual[j]);
    }
    
    lhs = -dual_temp - 1e-12;
    rhs = dual_local_max;
    
    for (i=0; i<m*p; i++) {
      del = LambdaT.data[i] - LambdaT_old.data[i];
      rhs += del*(gLambdaT.data[i] + 0.5*del/(*nu));
      dLambdaT.data[i] = del;
    }
    
    backtrack_count = 0;
    while (lhs > rhs && backtrack_count < 20) {
      // Reduce step-size
      *nu = 0.5*(*nu);
      
      // Increment backtrack_count
      backtrack_count++;
      
      // Copy Old Lambda^t into new Lambda^t
      for (i=0; i<m*p; i++)
	LambdaT.data[i] = LambdaT_old.data[i];
      
      // Redo Forward-Backward step
      update_LambdaT2(LambdaT, LambdaT_temp, gLambdaT, nu, w);
      
      // Compute Loss
      update_UT(XT, LambdaT, UT, Phi);
      convex_cluster_dual(XT, UT, &dual_temp);
      
      lhs = -dual_temp - 1e-12;
      rhs = dual_local_max;
      
      for (i=0; i<m*p; i++) {
	del = LambdaT.data[i] - LambdaT_old.data[i];
	rhs += del*(gLambdaT.data[i] + 0.5*del/(*nu));
	dLambdaT.data[i] = del;
      }
    }
    
    dual[its] = dual_temp;
    
    convex_cluster_primal(XT, UT, VT, Phi, w, &primal_temp);
    
    primal[its] = primal_temp;
    
    // Copy current gradient into last gradient
    for (i=0; i<m*p; i++)
      gLambdaT_old.data[i] = gLambdaT.data[i];
    // Compute next step-size using BB-spectral method
    grad_LambdaT(UT, Phi, gLambdaT);
    
    dLambdaSq = 0.0;
    dLambdaDotdGrad = 0.0;
    dGradSq = 0.0;
    for (i=0; i<m*p; i++) {
      dLambdaSq += dLambdaT.data[i]*dLambdaT.data[i];
      dg = gLambdaT.data[i] - gLambdaT_old.data[i];
      dLambdaDotdGrad += dLambdaT.data[i]*dg;
      dGradSq += dg*dg;
    }
    nu_s = dLambdaSq/dLambdaDotdGrad;
    nu_m = dLambdaDotdGrad/dGradSq;
    
    // Adaptive combination of nu_s and nu_m
    if (2.0*nu_m > nu_s)
      *nu = nu_m;
    else
      *nu = nu_s - 0.5*nu_m;
    
    if (primal_temp - dual_temp < *tol) break;
  }
  
  if (its == *max_iter)
    its -= 1;
  *iter = its;
}

/*
 Cobra: DLPA wrapper around Convex Clustering
*/
void convex_bicluster_dlpa(matrix XT,
			   matrix LambdaT_row,
			   matrix LambdaT_temp_row, matrix LambdaT_old_row, matrix dLambdaT_row,
			   matrix gLambdaT_row, matrix gLambdaT_old_row,
			   matrix LambdaT_col,
			   matrix LambdaT_temp_col, matrix LambdaT_old_col, matrix dLambdaT_col,
			   matrix gLambdaT_col, matrix gLambdaT_old_col,
			   matrix VT_row, matrix VT_col,
			   matrix UT, matrix YT, matrix PT, matrix QT,
			   spmatrix Phi_row, spmatrix Phi_col,
			   double *w_row, double *w_col,
			   double *nu_row, double *nu_col,
			   double *primal_row, double *dual_row,
			   double *primal_col, double *dual_col,
			   int *max_iter, int *iter, double *tol) {
  int its, i, j;
  int m_row, m_col, n, p;
  int max_iter_row, max_iter_col, iter_col, iter_row;
  double diff;
  matrix UP, YQ;
  double *primal_row_local, *dual_row_local;
  double *primal_col_local, *dual_col_local;
  
  // X is p-by-n
  m_row = Phi_row.Nrow;
  m_col = Phi_col.Nrow;
  n = Phi_col.Ncol;
  p = Phi_row.Ncol;
  
  // UP = U + P is p-by-n
  UP.Nrow = p;
  UP.Ncol = n;
  UP.data = (double*) calloc(n*p,sizeof(double));
  
  // YQ = Y + Q is n-by-p
  YQ.Nrow = n;
  YQ.Ncol = p;
  YQ.data = (double*) calloc(n*p,sizeof(double));
  
  // Initialize P^t = Q^t = 0
  for (i=0; i<n*p; i++) {
    PT.data[i] = 0.;
    QT.data[i] = 0.;
  }
    
  max_iter_row = 1000;
  max_iter_col = 1000;
  
  primal_row_local = (double*) calloc(max_iter_row,sizeof(double));
  dual_row_local = (double*) calloc(max_iter_row,sizeof(double));
  primal_col_local = (double*) calloc(max_iter_col,sizeof(double));
  dual_col_local = (double*) calloc(max_iter_col,sizeof(double));
  
  // Initialize: U^t <- X^t
  for (i=0; i<n*p; i++)
    UT.data[i] = XT.data[i];
  
  // Main loop
  for (its=0; its<*max_iter; its++) {
    // UP is p-by-n
    // UT, PT are n-by-p
    for (i=0; i<p; i++)
      for (j=0; j<n; j++)
	UP.data[i + p*j] = UT.data[j + n*i] + PT.data[j + n*i];
    
    // cluster p rows of UP = U + P
    convex_cluster_fasta(UP, YT, VT_row, LambdaT_row, LambdaT_temp_row,
			 LambdaT_old_row, dLambdaT_row,
			 gLambdaT_row, gLambdaT_old_row,
			 Phi_row, w_row, nu_row,
			 primal_row_local, dual_row_local, &max_iter_row, &iter_row, tol);
    
    update_VT_row(YT, LambdaT_row, VT_row, Phi_row, w_row, nu_row);

    // YT, QT are p-by-n
    // YQ, UT, PT are n-by-p
    for (i=0; i<n; i++)
      for (j=0; j<p; j++) {
	PT.data[i + n*j] += UT.data[i + n*j] - YT.data[j + p*i];
	YQ.data[i + n*j] = YT.data[j + p*i] + QT.data[j + p*i];
      }
    
    // cluster n columns of YQ^t = (Y + Q)^t
    convex_cluster_fasta(YQ, UT, VT_col, LambdaT_col, LambdaT_temp_col,
			 LambdaT_old_col, dLambdaT_col,
			 gLambdaT_col, gLambdaT_old_col,
			 Phi_col, w_col, nu_col,
			 primal_col_local, dual_col_local, &max_iter_col, &iter_col, tol);
    
    update_VT_col(UT, LambdaT_col, VT_col, Phi_col, w_col, nu_col);

    for (i=0; i<p; i++)
      for (j=0; j<n; j++)
	QT.data[i + p*j] += YT.data[i + p*j] - UT.data[j + n*i];
    
    // Compute discrepancy between Y and U.
    diff = 0.;
    for (i=0; i<n; i++)
      for (j=0; j<p; j++)
	diff += pow(UT.data[i + n*j] - YT.data[j + p*i], 2.);
    diff = sqrt(diff);
    
    // Record objective values.
    primal_row[its] = primal_row_local[iter_row];
    dual_row[its] = dual_row_local[iter_row];
    primal_col[its] = primal_col_local[iter_col];
    dual_col[its] = dual_col_local[iter_col];
    
    if (diff < (*tol)*n*p) break;
  }
  
  if (its < *max_iter)
    its += 1;
  *iter = its;
  //  update_V(U,Lambda_col,V_col,w_col,gamma,nu_col,ix_col,n,nK_col);
  //  update_V(Y,Lambda_row,V_row,w_row,gamma,nu_row,ix_row,p,nK_row);
  
  free(primal_row_local);
  free(dual_row_local);
  free(primal_col_local);
  free(dual_col_local);
  
  free(UP.data);
  free(YQ.data);
}

/*
 Test convex_bicluster_dlpa
*/
void test_convex_bicluster_dlpa(double *xt,
				double *lambdat_row,
				double *lambdat_temp_row, double *lambdat_old_row, double *dlambdat_row,
				double *glambdat_row, double *glambdat_old_row,
				double *lambdat_col,
				double *lambdat_temp_col, double *lambdat_old_col, double *dlambdat_col,
				double *glambdat_col, double *glambdat_old_col,
				double *vt_row, double *vt_col,
				double *ut, double *yt, double *pt, double *qt,
				int *column_ptr_row, int *row_indices_row, double *values_row,
				int *column_ptr_col, int *row_indices_col, double *values_col,
				int *m_row, int *m_col, int *n, int *p,
				double *w_row, double *w_col,
				double *nu_row, double *nu_col,
				double *primal_row, double *dual_row,
				double *primal_col, double *dual_col,
				int *max_iter, int *iter, double *tol) {
  matrix XT;
  matrix LambdaT_row, LambdaT_temp_row, LambdaT_old_row, dLambdaT_row, gLambdaT_row, gLambdaT_old_row;
  matrix LambdaT_col, LambdaT_temp_col, LambdaT_old_col, dLambdaT_col, gLambdaT_col, gLambdaT_old_col;
  matrix VT_row, VT_col;
  matrix UT, YT, PT, QT;
  spmatrix Phi_row, Phi_col;
  
  XT.Nrow = *n;
  XT.Ncol = *p;
  XT.data = xt;
  
  LambdaT_row.Nrow = *m_row;
  LambdaT_row.Ncol = *n;
  LambdaT_row.data = lambdat_row;
  
  LambdaT_temp_row.Nrow = *m_row;
  LambdaT_temp_row.Ncol = *n;
  LambdaT_temp_row.data = lambdat_temp_row;
  
  LambdaT_old_row.Nrow = *m_row;
  LambdaT_old_row.Ncol = *n;
  LambdaT_old_row.data = lambdat_old_row;
  
  dLambdaT_row.Nrow = *m_row;
  dLambdaT_row.Ncol = *n;
  dLambdaT_row.data = dlambdat_row;
  
  gLambdaT_row.Nrow = *m_row;
  gLambdaT_row.Ncol = *n;
  gLambdaT_row.data = glambdat_row;
  
  gLambdaT_old_row.Nrow = *m_row;
  gLambdaT_old_row.Ncol = *n;
  gLambdaT_old_row.data = glambdat_old_row;
  
  LambdaT_col.Nrow = *m_col;
  LambdaT_col.Ncol = *p;
  LambdaT_col.data = lambdat_col;
    
  LambdaT_temp_col.Nrow = *m_col;
  LambdaT_temp_col.Ncol = *p;
  LambdaT_temp_col.data = lambdat_temp_col;
  
  LambdaT_old_col.Nrow = *m_col;
  LambdaT_old_col.Ncol = *p;
  LambdaT_old_col.data = lambdat_old_col;
  
  dLambdaT_col.Nrow = *m_col;
  dLambdaT_col.Ncol = *p;
  dLambdaT_col.data = dlambdat_col;
  
  gLambdaT_col.Nrow = *m_col;
  gLambdaT_col.Ncol = *p;
  gLambdaT_col.data = glambdat_col;
  
  gLambdaT_old_col.Nrow = *m_col;
  gLambdaT_old_col.Ncol = *p;
  gLambdaT_old_col.data = glambdat_old_col;
  
  VT_row.Nrow = *m_row;
  VT_row.Ncol = *n;
  VT_row.data = vt_row;
  
  VT_col.Nrow = *m_col;
  VT_col.Ncol = *p;
  VT_col.data = vt_col;
  
  UT.Nrow = *n;
  UT.Ncol = *p;
  UT.data = ut;
  
  PT.Nrow = *n;
  PT.Ncol = *p;
  PT.data = pt;
  
  YT.Nrow = *p;
  YT.Ncol = *n;
  YT.data = yt;
  
  QT.Nrow = *p;
  QT.Ncol = *n;
  QT.data = qt;
  
  Phi_row.Nrow = *m_row;
  Phi_row.Ncol = *p;
  Phi_row.column_ptr = column_ptr_row;
  Phi_row.row_indices = row_indices_row;
  Phi_row.values = values_row;
  
  Phi_col.Nrow = *m_col;
  Phi_col.Ncol = *n;
  Phi_col.column_ptr = column_ptr_col;
  Phi_col.row_indices = row_indices_col;
  Phi_col.values = values_col;
  
  convex_bicluster_dlpa(XT, 
			LambdaT_row, 
			LambdaT_temp_row, LambdaT_old_row, dLambdaT_row,
			gLambdaT_row, gLambdaT_old_row,
			LambdaT_col, 
			LambdaT_temp_col, LambdaT_old_col, dLambdaT_col,
			gLambdaT_col, gLambdaT_old_col,
			VT_row, VT_col,
			UT, YT, PT, QT,
			Phi_row, Phi_col, 
			w_row, w_col, 
			nu_row, nu_col,
			primal_row, dual_row, 
			primal_col, dual_col, 
			max_iter, iter, tol);
  
}

/*
  Update Majorization
*/
void update_majorization(matrix MT, matrix UT, int *Theta, int *nMissing) {
  int i;
  for (i=0; i<*nMissing; i++)
    MT.data[Theta[i]] = UT.data[Theta[i]];
}

/*
 Cobra-POD: MM wrapper around Convex Biclustering
*/
void convex_bicluster_impute(matrix MT, matrix UT,
			     matrix LambdaT_row, matrix LambdaT_col,
			     matrix VT_row, matrix VT_col,
			     spmatrix Phi_row, spmatrix Phi_col,
			     int *Theta, int *nMissing,
			     double *w_row, double *w_col,
			     double *nu_row, double *nu_col,
			     double *mm_loss,
			     int *max_iter, int *iter, double *tol,
			     int *max_iter_inner, double *tol_inner) {
  
  int i, j, its, iter_inner;
  int m_row, m_col, n, p;
  matrix LambdaT_temp_row, LambdaT_old_row, dLambdaT_row, gLambdaT_row, gLambdaT_old_row;
  matrix LambdaT_temp_col, LambdaT_old_col, dLambdaT_col, gLambdaT_col, gLambdaT_old_col;
  matrix YT, PT, QT, VT_temp_row, VT_temp_col;
  
  double *primal_row = (double*) calloc(*max_iter_inner,sizeof(double));
  double *dual_row = (double*) calloc(*max_iter_inner,sizeof(double));
  double *primal_col = (double*) calloc(*max_iter_inner,sizeof(double));
  double *dual_col = (double*) calloc(*max_iter_inner,sizeof(double));
  double mm_loss_temp;
  double mm_loss_last;
  
  // Get matrix dimensions
  m_row = Phi_row.Nrow;
  m_col = Phi_col.Nrow;
  n = Phi_col.Ncol;
  p = Phi_row.Ncol;
  
  // Initialize nuisance parameters
  YT.Nrow = p;
  YT.Ncol = n;
  YT.data = (double*) calloc(p*n,sizeof(double));

  PT.Nrow = n;
  PT.Ncol = p;
  PT.data = (double*) calloc(n*p,sizeof(double));

  QT.Nrow = p;
  QT.Ncol = n;
  QT.data = (double*) calloc(p*n,sizeof(double));
    
  VT_temp_row.Nrow = m_row;
  VT_temp_row.Ncol = n;
  VT_temp_row.data = (double*) calloc(m_row*n,sizeof(double));

  VT_temp_col.Nrow = m_col;
  VT_temp_col.Ncol = p;
  VT_temp_col.data = (double*) calloc(m_col*p,sizeof(double));

  LambdaT_temp_row.Nrow = m_row;
  LambdaT_temp_row.Ncol = n;
  LambdaT_temp_row.data = (double*) calloc(m_row*n,sizeof(double));
  
  LambdaT_old_row.Nrow = m_row;
  LambdaT_old_row.Ncol = n;
  LambdaT_old_row.data = (double*) calloc(m_row*n,sizeof(double));
  
  dLambdaT_row.Nrow = m_row;
  dLambdaT_row.Ncol = n;
  dLambdaT_row.data = (double*) calloc(m_row*n,sizeof(double));
  
  gLambdaT_row.Nrow = m_row;
  gLambdaT_row.Ncol = n;
  gLambdaT_row.data = (double*) calloc(m_row*n,sizeof(double));
  
  gLambdaT_old_row.Nrow = m_row;
  gLambdaT_old_row.Ncol = n;
  gLambdaT_old_row.data = (double*) calloc(m_row*n,sizeof(double));
  
  LambdaT_temp_col.Nrow = m_col;
  LambdaT_temp_col.Ncol = p;
  LambdaT_temp_col.data = (double*) calloc(m_col*p,sizeof(double));
  
  LambdaT_old_col.Nrow = m_col;
  LambdaT_old_col.Ncol = p;
  LambdaT_old_col.data = (double*) calloc(m_col*p,sizeof(double));
  
  dLambdaT_col.Nrow = m_col;
  dLambdaT_col.Ncol = p;
  dLambdaT_col.data = (double*) calloc(m_col*p,sizeof(double));
  
  gLambdaT_col.Nrow = m_col;
  gLambdaT_col.Ncol = p;
  gLambdaT_col.data = (double*) calloc(m_col*p,sizeof(double));
  
  gLambdaT_old_col.Nrow = m_col;
  gLambdaT_old_col.Ncol = p;
  gLambdaT_old_col.data = (double*) calloc(m_col*p,sizeof(double));
  
  // Initial majorization
  update_majorization(MT, UT, Theta, nMissing);

  // Compute primal loss
  convex_bicluster_primal(MT, UT, VT_temp_row, VT_temp_col, Phi_row, Phi_col, w_row, w_col, &mm_loss_last);

  mm_loss[0] = mm_loss_last;

  // Main loop
  for (its=1; its<*max_iter; its++) {
  
    convex_bicluster_dlpa(MT,
			  LambdaT_row,
			  LambdaT_temp_row, LambdaT_old_row, dLambdaT_row,
			  gLambdaT_row, gLambdaT_old_row,
			  LambdaT_col,
			  LambdaT_temp_col, LambdaT_old_col, dLambdaT_col,
			  gLambdaT_col, gLambdaT_old_col,
			  VT_row, VT_col,
			  UT, YT, PT, QT,
			  Phi_row, Phi_col,
			  w_row, w_col,
			  nu_row, nu_col,
			  primal_row, dual_row,
			  primal_col, dual_col,
			  max_iter_inner, &iter_inner, tol_inner);

    update_majorization(MT, UT, Theta, nMissing);

    // Compute primal loss
    convex_bicluster_primal(MT, UT, VT_temp_row, VT_temp_col, Phi_row, Phi_col, w_row, w_col, &mm_loss_temp);
    
    mm_loss[its] = mm_loss_temp;
    
    if (mm_loss_last >= mm_loss_temp)
      if ((mm_loss_last - mm_loss_temp) < (*tol)*(1. + mm_loss_last)) break;
    
    mm_loss_last = mm_loss_temp;

  }
  
  if (its < *max_iter)
    its += 1;
  *iter = its;
  
  free(YT.data);
  free(PT.data);
  free(QT.data);
  free(VT_temp_row.data);
  free(VT_temp_col.data);
  
  free(LambdaT_temp_row.data);
  free(LambdaT_old_row.data);
  free(dLambdaT_row.data);
  free(gLambdaT_row.data);
  free(gLambdaT_old_row.data);
  
  free(LambdaT_temp_col.data);
  free(LambdaT_old_col.data);
  free(dLambdaT_col.data);
  free(gLambdaT_col.data);
  free(gLambdaT_old_col.data);
  
  free(primal_row);
  free(dual_row);
  free(primal_col);
  free(dual_col);
}

/*
  Test convex_bicluster_impute
*/
void test_convex_bicluster_impute(double *mt, double *ut,
				  double *lambdat_row, double *lambdat_col,
				  double *vt_row, double *vt_col,
				  int *column_ptr_row, int *row_indices_row, double *values_row,
				  int *column_ptr_col, int *row_indices_col, double *values_col,
				  int *m_row, int *m_col, int *n, int *p,
				  int *Theta, int *nMissing,
				  double *w_row, double *w_col,
				  double *nu_row, double *nu_col,
				  double *mm_loss,
				  int *max_iter, int *iter, double *tol,
				  int *max_iter_inner, double *tol_inner) {
  
  matrix MT, UT;
  matrix LambdaT_row, LambdaT_col;
  matrix VT_row, VT_col;
  spmatrix Phi_row, Phi_col;
  
  MT.Nrow = *n;
  MT.Ncol = *p;
  MT.data = mt;
  
  UT.Nrow = *n;
  UT.Ncol = *p;
  UT.data = ut;
  
  LambdaT_row.Nrow = *m_row;
  LambdaT_row.Ncol = *n;
  LambdaT_row.data = lambdat_row;
  
  LambdaT_col.Nrow = *m_col;
  LambdaT_col.Ncol = *p;
  LambdaT_col.data = lambdat_col;
  
  VT_row.Nrow = *m_row;
  VT_row.Ncol = *n;
  VT_row.data = vt_row;
  
  VT_col.Nrow = *m_col;
  VT_col.Ncol = *p;
  VT_col.data = vt_col;
  
  Phi_row.Nrow = *m_row;
  Phi_row.Ncol = *p;
  Phi_row.column_ptr = column_ptr_row;
  Phi_row.row_indices = row_indices_row;
  Phi_row.values = values_row;
    
  Phi_col.Nrow = *m_col;
  Phi_col.Ncol = *n;
  Phi_col.column_ptr = column_ptr_col;
  Phi_col.row_indices = row_indices_col;
  Phi_col.values = values_col;
    
  convex_bicluster_impute(MT, UT, LambdaT_row, LambdaT_col,
                          VT_row, VT_col,
                          Phi_row, Phi_col,
                          Theta, nMissing,
                          w_row, w_col, 
                          nu_row, nu_col,
                          mm_loss,
                          max_iter, iter, tol,
                          max_iter_inner, tol_inner);
}
