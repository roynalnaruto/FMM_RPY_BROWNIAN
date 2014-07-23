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

int interactions(int numpos, double *pos, double L, int boxdim, double cutoff2, double *distances2, int *pairs, int maxnumpairs, int *numpairs_p){
    
    
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
    int *next = (int *) malloc(numpos*sizeof(int));
    if (next == NULL)
    {
        printf("interactions: could not malloc array for %d particles\n", numpos);
        return 1;
    }
    

    // traverse all particles and assign to boxes
    int i;
    for (i=0; i<numpos; i++)
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
                        
                        if ((d2 = dx*dx+dy*dy+dz*dz) <= cutoff2)
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

					//cout<<"done allocating neigh_id"<<endl;

                    neigh_bp = &b[neigh_idx][neigh_idy][neigh_idz];
                    
                    //cout<<"hey 1"<<endl;

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
                    
                    //cout<<"hey 2"<<endl;

                    p1 = neigh_bp->head;
                    //cout<<"entering WHILE"<<endl;
                    while (p1 != -1)
                    {
						//cout<<"still p1 there"<<endl;
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
	printf("Total number of overlapping pairs : %d\n", numpairs);

    return 0;
}

void interactionsFilter(int npos, int *numpairs_p, int *pairs, int *finalPairs, double *rad, double *pos){

    int kk, i, interactingPairs = 0;
    double r[3];
    double actualDistance, distance;
    int nonshellPairs = 0;

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
            finalPairs[2*interactingPairs] = min(pairs[i], pairs[i+1]);
            finalPairs[2*interactingPairs+1] = max(pairs[i], pairs[i+1]);
            interactingPairs++;
            //if((pairs[i] <= npos) && (pairs[i+1] <= npos))
			nonshellPairs++;
        }
        
    }

    *numpairs_p = interactingPairs;
    printf("Non shell particles , number of overlaps %d \n", nonshellPairs);
    printf("Total number of overlapping particles : %d\n", interactingPairs);
}




void computeForce(int npos, double *f, double *pos, double *rad, int *pairs, int numpairs){

    int i, j, pair;
    double a1, a2;
    double r1, r2, r3, s;

    //initialize to zero
    for(i=0; i<npos; i++){
        f[3*i + 0] = 0.0;
        f[3*i + 1] = 0.0;
        f[3*i + 2] = 0.0;
    }
    
    int count=0;

    for(pair=0; pair<2*numpairs; pair+=2){

        i = pairs[pair]-1;
        j = pairs[pair+1]-1;

        a1 = rad[i];
        r1 = pos[3*i + 0] - pos[3*j + 0];
        r2 = pos[3*i + 1] - pos[3*j + 1];
        r3 = pos[3*i + 2] - pos[3*j + 2];
        s = sqrt(r1*r1 + r2*r2 + r3*r3);

        if(j >= npos){
            f[3*i + 0] += -kshell*(1 - (a1 + shell_particle_radius)/s)*r1;
            f[3*i + 1] += -kshell*(1 - (a1 + shell_particle_radius)/s)*r2;
            f[3*i + 2] += -kshell*(1 - (a1 + shell_particle_radius)/s)*r3;
		}
        else{
			count++;
			a2 = rad[j];
            f[3*i + 0] += -kparticle*(1 - (a1 + a2)/s)*r1;
            f[3*i + 1] += -kparticle*(1 - (a1 + a2)/s)*r2;
            f[3*i + 2] += -kparticle*(1 - (a1 + a2)/s)*r3;
            f[3*j + 0] += kparticle*(1 - (a1 + a2)/s)*r1;
            f[3*j + 1] += kparticle*(1 - (a1 + a2)/s)*r2;
            f[3*j + 2] += kparticle*(1 - (a1 + a2)/s)*r3;
			}

		}	 
//	cout<<"NON_SHELL OVERLAPPING PARTICLES COUNT: "<<count<<endl;	
}







void computeForceSerial(int npos, int nsphere, double *f, double *pos, double *rad, double *shell){
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
    int npairs = 0;

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
				npairs ++;
                for(m=0; m<3; m++){
                    f[3*i+m] += -kparticle*(1 - (a1+a2)/s)*r[m];
                }
            }
        }
    }
    assert(npairs%2==0);
    printf("exit computeForce: npairs  = %d \n", npairs/2);
}

