#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "slp_joint.hpp"
#include "tools.hpp"


int main(int argc, char **argv)
{ 
	int i;
	double l1, l2;
	struct timeval tim;
	int rep = 1000;

    int N = 160;
    int n = 2*N-1;
    int K[] = {10, 30, 50, 100};
    int Kl = 4;
    int k;
    bool fftfiltering = true;

    FTYPE * xk = vector(n);
    FTYPE * x = vector(2*N);

    slp_joint_iir * slp = new slp_joint_iir(N, x, K[0], -1, 
                                            0.3, 0.4, fftfiltering);

    printf("-- SLP_JOINT_IIR -- \n");
    printf(" fftfiltering = %s \n", fftfiltering ? "true" : "false");
    for( k = 0 ; k < Kl ; k++){
      slp->set_kmax(K[k]);
      /* warm up */
      for( i = 0 ; i < 100; ++i ){
        CvmdInit(n, 0, xk);
        slp->update_signal(x);
        ofm(slp, xk);
      }
      
      /* run timing */
      gettimeofday(&tim, NULL);
      l1 = tim.tv_sec + (tim.tv_usec/1000000.0);
      for( i = 0 ; i < rep; ++i ){
        CvmdInit(n, 0, xk);
        slp->update_signal(x);
        ofm(slp, xk);
      }
      gettimeofday(&tim, NULL);
      l2 = tim.tv_sec + (tim.tv_usec/1000000.0);

      printf("Kmax = %d, N = %d, repetitions = %d \n", K[k], N, rep);
      printf("Average time : %2.5f [ms]\n", 1000*(l2-l1)/rep);
      printf("\n");
    }

    del_vector(xk);
    delete slp;

}
