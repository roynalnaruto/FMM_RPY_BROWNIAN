#include "headers.h"

inline int min(int a, int b){
    return ((a>b)? b : a);
}

inline int max(int a, int b){
    return ((a>b)? a : b);
}

void printPairs(int *array, int pair){

    int i;
    printf("\n");
    for(i=0; i<pair; i+=2){
        printf("%d, %d\n", array[i], array[i+1]);
    }
    printf("\n");
}

void printVectors(double *array, int size, int dimension){
    int i,j;
    printf("\n");
    for(i=0;i<size;i++){
        for(j=0;j<dimension;j++)
            printf("%lf\t", array[3*i + j]);
        printf("\n");
    }
    printf("\n");
}

void setPosRad(double *pos, double *rad){
    //printf("enter setPosRad\n");
    double temp_r, temp_theta, temp_phi;
    int i, j;
    for(i=0; i<npos; i++){

        //r, phi and theta
        temp_r = (double)rand()/(double)(RAND_MAX/1.0);
        temp_r = pow(temp_r, 1.0/3.0)*(shell_radius - 1.0);
        temp_theta = -1.0 + (float)rand()/(float)(RAND_MAX/2.0);
        temp_theta = acos(temp_theta);
        temp_phi = (float)rand()/(float)(RAND_MAX/(2.0*pie));

        rad[i] = 0.3 + (double)rand()/(double)(RAND_MAX/0.7);
        //spherical co-ordinates to cartesian
        pos[0+(i*3)] = temp_r*sin(temp_theta)*cos(temp_phi) + shell_radius;
        pos[1+(i*3)] = temp_r*sin(temp_theta)*sin(temp_phi) + shell_radius;
        pos[2+(i*3)] = temp_r*cos(temp_theta) + shell_radius;

    }

    //printf("exit setPosRad\n");
}

void getShell(double *shell){
    //printf("enter getShell\n");
    char input_index[100];
    int i, j;
    FILE *input;
    sprintf(input_index, "/home/rohit/final/data-for-outer-shell/data%d_%d.csv", nsphere, shell_radius);
    input = fopen(input_index, "r");
    for(i=0; i<nsphere; i++){
        for(j=0; j<3; j++){
            fscanf(input, "%lf, ", &shell[3*i + j]);
            shell[3*i + j] += shell_radius;
        }
        fscanf(input, "\n");
    }
    fclose(input);
    //printf("exit getShell\n");
}





////////////////////////////////////////////////////////////
void computeForceSerial(double *f, double *pos, double *rad, double *shell){
    //printf("enter computeForce\n");

    //initialize variables
    double a1, a2;
    int k, m, n, i, j;
    double s;
    double pi[3], pj[3];
    double r[3];

    double r1, r2, r3;
    //float f1, f2, f3;

    double f_shell[npos*3];

    //force between outer shell and inner particles
    for(i=0; i<npos; i++){

        a1 = rad[i];
        f_shell[3*i+0] = f_shell[3*i+1] = f_shell[3*i+2] = 0.0;

        for(j=0; j<nsphere; j++){

            r1 = pos[3*i+0] - shell[3*j+0];
            r2 = pos[3*i+1] - shell[3*j+1];
            r3 = pos[3*i+2] - shell[3*j+2];
            s = sqrt(r1*r1 + r2*r2 + r3*r3);
            //printf("%f\n", s);

            if(s<(a1+shell_particle_radius)){
                f_shell[3*i+0] += -kshell*(1 - (a1+shell_particle_radius)/s)*r1;
                f_shell[3*i+1] += -kshell*(1 - (a1+shell_particle_radius)/s)*r2;
                f_shell[3*i+2] += -kshell*(1 - (a1+shell_particle_radius)/s)*r3;
            }
            //printf("%f\n", f_shell[i][0] + f_shell[i][1] + f_shell[i][2]);

        }

    }

    //loop
    for(i=0; i<npos; i++){

        //initialize for particle i
        //f[3*i] = f[3*i+1] = f[3*i+2] = 0.0;
        f[3*i + 0] = f_shell[3*i+0];
        f[3*i + 1] = f_shell[3*i+1];
        f[3*i + 2] = f_shell[3*i+2];

        //interact with all other particles j (j != i)
        for(j=0; j<npos; j++){

            if(i==j)
                continue;

            //create pi and pj and compute r
            for(k=0; k<3; k++){
                pi[k] = pos[3*i+k];
                pj[k] = pos[3*j+k];
                r[k] = pi[k] - pj[k];
            }

            //euclidean distance
            s = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

            //get radii
            a1 = rad[i];
            a2 = rad[j];

            //compute forces if s<=(a1+a2)
            if(s<=(a1+a2)){
                for(m=0; m<3; m++){
                    f[3*i+m] += -kparticle*(1 - (a1+a2)/s)*r[m];
                }
            }
        }
    }

    //printf("exit computeForce\n");
}
////////////////////////////////////////////////////////////