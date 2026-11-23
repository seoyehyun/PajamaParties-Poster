// membw_probe.c -- streaming memory probe with a known byte count, for calibrating PMU counters (see membw_calib.sh).
//
//   membw_probe <read|write|copy|triad> <threads> <iters> [MiB per array, default 512]
//
// Prints one CSV line on stdout: pattern,threads,iters,loop_seconds,bytes_per_iter
// bytes_per_iter is what the program streams in one iteration (read 1 array, write 1, copy 2, triad 3).
// loop_seconds covers only the iterations (not the first-touch initialisation).
// Build: gcc -O3 -ffast-math -fopenmp -o membw_probe membw_probe.c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s read|write|copy|triad threads iters [MiB]\n", argv[0]); return 2; }
  const char *pat = argv[1];
  int nt = atoi(argv[2]), iters = atoi(argv[3]);
  size_t mib = argc > 4 ? strtoull(argv[4], 0, 10) : 512, N = mib * (1u << 20) / 8;
  int kind = !strcmp(pat, "read") ? 0 : !strcmp(pat, "write") ? 1 : !strcmp(pat, "copy") ? 2 : !strcmp(pat, "triad") ? 3 : -1;
  if (kind < 0) { fprintf(stderr, "unknown pattern %s\n", pat); return 2; }
  double *a = aligned_alloc(64, N * 8), *b = aligned_alloc(64, N * 8), *c = aligned_alloc(64, N * 8);
  if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
  omp_set_num_threads(nt);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }

  double sink = 0, t0 = omp_get_wtime();
  for (int k = 0; k < iters; k++) {
    if (kind == 0) {
      double s = 0;
      #pragma omp parallel for schedule(static) reduction(+:s)
      for (size_t i = 0; i < N; i++) s += b[i];
      sink += s;
    } else if (kind == 1) {
      double v = k + 1;
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; i++) a[i] = v;
    } else if (kind == 2) {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; i++) a[i] = b[i];
    } else {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; i++) a[i] = b[i] + 3.0 * c[i];
    }
  }
  double t1 = omp_get_wtime();
  static const int arrays[4] = {1, 1, 2, 3};
  printf("%s,%d,%d,%.6f,%.0f\n", pat, nt, iters, t1 - t0, (double)arrays[kind] * 8.0 * (double)N);
  fprintf(stderr, "sink=%g a=%g\n", sink, a[N / 2]);   // keeps the loops alive
  return 0;
}
