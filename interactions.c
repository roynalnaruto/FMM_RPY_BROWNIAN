#include "headers.h"

int box_neighbors[NUM_BOX_NEIGHBORS][3] =
{
    {-1,-1,-1},
    {-1,-1, 0},
    {-1,-1,+1},
    {-1, 0,-1},
    {-1, 0, 0},
    {-1, 0,+1},
    {-1,+1,-1},
    {-1,+1, 0},
    {-1,+1,+1},
    { 0,-1,-1},
    { 0,-1, 0},
    { 0,-1,+1},
    { 0, 0,-1}
};

int interactions(int npos, double *pos, double L, int boxdim, double cutoff2, double *distances2, int *pairs, int maxnumpairs, int *numpairs_p){
    


    if (boxdim < 4 || cutoff2 > (L/boxdim)*(L/boxdim))
    {
        printf("interactions: bad input parameters\n");
        return 1;
    }

    struct box b[boxdim][boxdim][boxdim];
    struct box *bp;
    struct box *neigh_bp;

    // box indices
    int idx, idy, idz;
    int neigh_idx, neigh_idy, neigh_idz;

    // allocate memory for particles in each box
    for (idx=0; idx<boxdim; idx++)
    for (idy=0; idy<boxdim; idy++)
    for (idz=0; idz<boxdim; idz++)
        b[idx][idy][idz].head = -1;

    // allocate implied linked list
    int *next = (int *) malloc(npos*sizeof(int));
    if (next == NULL)
    {
        printf("interactions: could not malloc array for %d particles\n", npos);
        return 1;
    }

    // traverse all particles and assign to boxes
    int i;
    for (i=0; i<npos; i++)
    {
        // initialize entry of implied linked list
        next[i] = -1;

        // which box does the particle belong to?
        // assumes particles have positions within [0,L]^3
        idx = (int)(pos[3*i  ]/L*boxdim);
        idy = (int)(pos[3*i+1]/L*boxdim);
        idz = (int)(pos[3*i+2]/L*boxdim);

        // add to beginning of implied linked list
        bp = &b[idx][idy][idz];
        next[i] = bp->head;
        bp->head = i;
    }

    int numpairs = 0;
    int p1, p2;
    double d2, dx, dy, dz;

    for (idx=0; idx<boxdim; idx++)
    {
        for (idy=0; idy<boxdim; idy++)
        {
            for (idz=0; idz<boxdim; idz++)
            {
                bp = &b[idx][idy][idz];

                // within box interactions
                p1 = bp->head;
                while (p1 != -1)
                {
                    p2 = next[p1];
                    while (p2 != -1)
                    {
                        if (numpairs >= maxnumpairs)
                            return -1;

                        // do not need minimum image since we are in same box
                        dx = pos[3*p1+0] - pos[3*p2+0];
                        dy = pos[3*p1+1] - pos[3*p2+1];
                        dz = pos[3*p1+2] - pos[3*p2+2];

                        if ((d2 = dx*dx+dy*dy+dz*dz) < cutoff2)
                        {
                            pairs[2*numpairs]   = p1 + 1;
                            pairs[2*numpairs+1] = p2 + 1;
                            distances2[numpairs] = d2;
                            numpairs++;
                        }

                        p2 = next[p2];
                    }
                    p1 = next[p1];
                }

                // interactions with other boxes
                int j;
                for (j=0; j<NUM_BOX_NEIGHBORS; j++)
                {
                    neigh_idx = (idx + box_neighbors[j][0] + boxdim) % boxdim;
                    neigh_idy = (idy + box_neighbors[j][1] + boxdim) % boxdim;
                    neigh_idz = (idz + box_neighbors[j][2] + boxdim) % boxdim;

                    neigh_bp = &b[neigh_idx][neigh_idy][neigh_idz];

                    // when using boxes, the minimum image computation is 
                    // known beforehand, thus we can  compute position offsets 
                    // to compensate for wraparound when computing distances
                    double xoffset = 0.;
                    double yoffset = 0.;
                    double zoffset = 0.;
                    if (idx + box_neighbors[j][0] == -1)     xoffset = -L;
                    if (idy + box_neighbors[j][1] == -1)     yoffset = -L;
                    if (idz + box_neighbors[j][2] == -1)     zoffset = -L;
                    if (idx + box_neighbors[j][0] == boxdim) xoffset =  L;
                    if (idy + box_neighbors[j][1] == boxdim) yoffset =  L;
                    if (idz + box_neighbors[j][2] == boxdim) zoffset =  L;

                    p1 = neigh_bp->head;
                    while (p1 != -1)
                    {
                        p2 = bp->head;
                        while (p2 != -1)
                        {
                            if (numpairs >= maxnumpairs)
                                return -1;

                            // compute distance vector
                            dx = pos[3*p1+0] - pos[3*p2+0] + xoffset;
                            dy = pos[3*p1+1] - pos[3*p2+1] + yoffset;
                            dz = pos[3*p1+2] - pos[3*p2+2] + zoffset;

                            if ((d2 = dx*dx+dy*dy+dz*dz) < cutoff2)
                            {
                                pairs[2*numpairs]   = p1 + 1;
                                pairs[2*numpairs+1] = p2 + 1;
                                distances2[numpairs] = d2;
                                numpairs++;
                            }

                            p2 = next[p2];
                        }
                        p1 = next[p1];
                    }
                }
            }
        }
    }

    free(next);

    *numpairs_p = numpairs;

    return 0;
}

void interactionsFilter(int *numpairs_p, int *pairs, int *finalPairs, double *rad, double *pos){

    int kk, i, interactingPairs = 0;
    double r[3];
    double actualDistance, distance;

    for(i=0; i<2*(*numpairs_p); i+=2){

        distance = 0;
        distance += (pairs[i] <= npos)? rad[pairs[i]-1] : shell_particle_radius;
        distance += (pairs[i+1] <= npos)? rad[pairs[i+1]-1] : shell_particle_radius;
        
        actualDistance = 0;
        
        for(kk=0;kk<3;kk++){
            r[kk] = pos[3*(pairs[i]-1) + kk] - pos[3*(pairs[i+1]-1) + kk];
            actualDistance += r[kk]*r[kk]; 
        }
        if((actualDistance <= distance * distance) && (pairs[i] <= npos || pairs[i+1] <= npos)){
            interactingPairs++;
            finalPairs[2*interactingPairs - 1] = pairs[i];
            finalPairs[2*interactingPairs] = pairs[i+1];
        }

    }

    *numpairs_p = interactingPairs;
}

void computeForce(double *f, complex *f1, complex *f2, complex *f3, double *pos, double *rad){

}