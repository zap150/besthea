#include <cstdio>


template<int x>
int funkce(int a);

template<>
int funkce<5>(int a)
{
    return 55*a;
}

template<>
int funkce<7>(int a)
{
    return 77*a;
}



int main()
{
    int a = 10;

    int y = funkce<5>(a);

    printf("y = %d\n", y);

    return 0;
}
