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

    besthea::tools::timer t;

    lo block_count = atoll(argv[1]);
    lo block_size = atoll(argv[2]);

    block_vector u1(block_count, block_size);
    block_vector u2(block_count, block_size);
    block_vector v_bv(block_count, block_size);
    sc * v_raw = new sc[v_bv.size()];

    t.reset("Setting");
    for(lo i = 0; i < block_count; i++) {
        for(lo j = 0; j < block_size; j++) {
            sc val = rand() % 100;
            u1.set(i, j, val);
            u2.set(i, j, val);

            val = rand() % 100;
            v_bv.set(i, j, val);
        }
    }
    t.measure();


    t.reset("Copying");
    v_bv.copy_to_raw(v_raw);
    t.measure();

    t.reset("Adding vector");
    u1.add(v_bv, 2);
    t.measure();

    t.reset("Adding raw");
    u2.add_from_raw(v_raw, 2);
    t.measure();
    


    t.reset("Checking");
    bool equal = true;
    for(lo i = 0; i < block_count; i++) {
        for(lo j = 0; j < block_size; j++) {
            sc val_1 = u1.get(i, j);
            sc val_2 = u2.get(i, j);
            if(std::abs((val_1 - val_2) / val_1) > 1e-10) {
                printf("WRONG I %ld J %ld vals %f %f\n", i, j, val_1, val_2);
                equal = false;
            }
        }
    }
    t.measure();



    if(equal) {
        printf("add_from_raw: correct!\n");
    }


    delete[] v_raw;

    return 0;
}
