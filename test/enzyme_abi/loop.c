// A time loop differentiated by Enzyme's LLVM core, whose checkpointing
// scheme is given by the caller (Checkpointing.jl, through enzyme_scheme).
#include <enzyme/checkpoint.h>
#include <math.h>

#define N 8

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

__attribute__((noinline)) static void step(int64_t i, double *u) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + 0.1 * (u[(k + 1) % N] - 2 * u[k] + u[(k + N - 1) % N]) +
             0.02 * sin(u[k] + 0.1 * (double)i);
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
}

static double loss(const double *u) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * u[k];
  return s;
}

static double plain(double *u, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u);
  return loss(u);
}

static double checkpointed(double *u, int64_t n,
                           const EnzymeCheckpointScheme *scheme, void *data) {
  __enzyme_checkpoint_for((void *)step, 0, n, enzyme_scheme, scheme, data,
                          enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u);
  return loss(u);
}

int64_t state_size(void) { return N; }

void gradient_plain(double *u, double *du, int64_t n) {
  __enzyme_autodiff((void *)plain, enzyme_dup, u, du, enzyme_const, n);
}

void gradient_checkpointed(double *u, double *du, int64_t n,
                           const EnzymeCheckpointScheme *scheme, void *data) {
  __enzyme_autodiff((void *)checkpointed, enzyme_dup, u, du, enzyme_const, n,
                    enzyme_const, scheme, enzyme_const, data);
}
