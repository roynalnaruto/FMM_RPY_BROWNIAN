typedef struct {double dr,di;} complex;

extern void computerpy_(int *n, double *source, double *radii, complex *v1, complex *v2, complex *v3, complex *rpy, char* dir);

extern void postcorrection_(int *i, int *j, double *source, complex *V1, complex *V2, complex *V3, 
						complex *rpy, int *nsource, double *radii, complex *C1);
