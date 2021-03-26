#include <cstdio>
#include <array>


constexpr int myFunc(int a)
{
    return 2 * a + 1;
}



struct Neco
{
    int x;

    template<int count>
    std::array<int, myFunc(count)> ToArray();
};


template<int count>
std::array<int, myFunc(count)> Neco::ToArray()
{
    constexpr int realCount = myFunc(count);
    std::array<int, realCount> arr;
    for(int i = 0; i < realCount; i++)
        arr[i] = x;
    
    return arr;
}


int main()
{
    Neco n;
    n.x = 42;

    constexpr int count = 7;
    constexpr int realCount = myFunc(count);
    std::array<int, realCount> arr = n.ToArray<count>();

    for(int i = 0; i < realCount; i++)
    {
        printf("%d: %d\n", i, arr[i]);
    }

    return 0;
}
