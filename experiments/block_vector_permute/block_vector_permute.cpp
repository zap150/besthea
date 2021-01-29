#include <cstdio>
#include <cstdlib>

#include "besthea/block_vector.h"
#include "besthea/timer.h"


using namespace besthea::linear_algebra;

int main(int argc, char ** argv)
{
    if(argc <= 2) {
        printf("Not enough arguments\n");
        return 0;
    }

    lo sz1 = atoll(argv[1]);
    lo sz2 = atoll(argv[2]);

    block_vector u(sz1, sz2);

    besthea::tools::timer t;

    t.reset("Setting");
    for(lo i = 0; i < u.get_block_size(); i++) {
        for(lo j = 0; j < u.get_size_of_block(); j++) {
            sc val = rand() % 100;
            u.set(i, j, val);
            //printf("%2d ", (int)val);
        }
        //printf("\n");
    }
    t.measure();

    block_vector v;

    t.reset("Permuting");
    v.copy_permute(u,2);
    t.measure();

    
    /*for(lo i = 0; i < v.get_block_size(); i++) {
        for(lo j = 0; j < v.get_size_of_block(); j++) {
            printf("%2d ", (int)v.get(i,j));
        }
        printf("\n");
    }*/

    block_vector w;
    w.copy_permute(v,0.5);

    bool equal = true;
    for(lo i = 0; i < u.get_block_size(); i++) {
        for(lo j = 0; j < u.get_size_of_block(); j++) {
            sc val_u = u.get(i, j);
            sc val_w = w.get(i, j);
            if(std::abs((val_u - val_w) / val_u) > 1e-10) {
                printf("WRONG I %ld J %ld vals %f %f\n", i, j, val_u, val_w);
                equal = false;
            }
        }        
    }
    if(equal) {
        printf("Correct!\n");
    }



    return 0;
}
