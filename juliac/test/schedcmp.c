// Drives a C reference scheme and the juliac-built one side by side, without
// AD: same actions, and a region stored in a slot comes back on restore.
#include <enzyme/checkpoint.h>
#include <stdio.h>
#include <string.h>

void *enzyme_ckpt_jl_revolve_scheme(void);
void *enzyme_ckpt_jl_revolve(int64_t);
void *enzyme_ckpt_jl_periodic_scheme(void);
void *enzyme_ckpt_jl_periodic(int64_t);
void enzyme_ckpt_jl_free(void *);

static int run(const EnzymeCheckpointScheme *A, void *da,
               const EnzymeCheckpointScheme *B, void *db, int64_t n) {
  double ra = 0, rb = 0;
  EnzymeCkptRegion rega = {&ra, sizeof ra, 0, 0}, regb = {&rb, sizeof rb, 0, 0};
  void *sa = A->init(da, n, sizeof ra), *sb = B->init(db, n, sizeof rb);
  int64_t k = 0;
  double held[256];
  for (;; k++) {
    EnzymeCkptAction a, b;
    A->next_action(sa, &a);
    B->next_action(sb, &b);
    if (a.flag != b.flag || a.iteration != b.iteration ||
        a.startiteration != b.startiteration || a.cpnum != b.cpnum) {
      printf("n=%ld action %ld: C (%d,%ld,%ld,%ld) jl (%d,%ld,%ld,%ld)\n", n, k,
             a.flag, a.iteration, a.startiteration, a.cpnum, b.flag,
             b.iteration, b.startiteration, b.cpnum);
      return 1;
    }
    if (a.flag == ENZYME_CKPT_DONE || a.flag == ENZYME_CKPT_ERROR)
      break;
    // The state before step `startiteration` is that number.
    if (a.flag == ENZYME_CKPT_STORE) {
      ra = rb = held[a.cpnum + 2] = (double)(a.startiteration + 1000 * k);
      A->store(sa, a.cpnum, a.startiteration, &rega, 1);
      B->store(sb, b.cpnum, b.startiteration, &regb, 1);
    } else if (a.flag == ENZYME_CKPT_RESTORE) {
      ra = rb = -1;
      A->restore(sa, a.cpnum, a.startiteration, &rega, 1);
      B->restore(sb, b.cpnum, b.startiteration, &regb, 1);
      if (ra != held[a.cpnum + 2] || rb != ra) {
        printf("n=%ld restore of slot %ld: C %g jl %g\n", n, a.cpnum, ra, rb);
        return 1;
      }
    }
  }
  A->finalize(sa);
  B->finalize(sb);
  return 0;
}

int main(void) {
  const EnzymeCheckpointScheme *jr = enzyme_ckpt_jl_revolve_scheme();
  const EnzymeCheckpointScheme *jp = enzyme_ckpt_jl_periodic_scheme();
  int fails = 0, runs = 0;
  for (int64_t c = 1; c <= 8; c++) {
    EnzymeCkptConfig cfg = {c, 0, NULL, 0, NULL};
    void *dr = enzyme_ckpt_jl_revolve(c), *dp = enzyme_ckpt_jl_periodic(c);
    for (int64_t n = 1; n <= 64; n++) {
      fails += run(&EnzymeCkptRevolve, &cfg, jr, dr, n);
      if (c <= n)
        fails += run(&EnzymeCkptPeriodic, &cfg, jp, dp, n);
      runs += 1 + (c <= n);
    }
    enzyme_ckpt_jl_free(dr);
    enzyme_ckpt_jl_free(dp);
  }
  printf("%d schedules, %d differ\n", runs, fails);
  return fails != 0;
}
