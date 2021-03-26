#include <cstdio>

#include "header.h"


int main()
{
    Neco<int> n(5);

    printf("Value is %d\n", n.Get());
    
    n.Set(42);

    printf("Value is %d\n", n.Get());
    



    return 0;
}
