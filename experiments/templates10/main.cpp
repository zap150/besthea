#include <cstdio>


template<int a, int b>
void Print()
{
    printf("%d %d\n", a, b);
}


// error: non-class, non-variable partial specialization ‘Print<42, b>’ is not allowed
/*
template<int b>
void Print<42, b>()
{
    printf("AAA 42 %d\n", b);
}
*/



int main()
{
    Print<31, 16>();

    //Print<42, 8>();



    return 0;
}

