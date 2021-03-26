#include <cstdio>


constexpr int some_func(int x)
{
    return 2 * x + 3;
}


template<int size>
struct my_array
{
    int data[size];
};

template<int other_size>
struct my_array_wrapper
{
    my_array<some_func(other_size)> arr;
};


int main()
{
    constexpr int s = 3;

    my_array_wrapper<s> maw;

    return 0;
}
