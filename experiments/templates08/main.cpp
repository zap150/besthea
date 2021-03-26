#include <cstdio>

template<int a, int b>
int function(int x)
{
    return x + 2 * a + 3 * b;
}

/*
template<int a>
int function<a, a>(int x)
{
    return x + 5 * a;
}*/


template<typename T1, typename T2>
struct Pair
{
    T1 x;
    T2 y;
};

template<typename T>
struct Pair<T,T>
{
    T x;
    T y;
};


int main()
{

    int y = function<7, 0>(11);

    printf("%d\n", y);


    return 0;
}
