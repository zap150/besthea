#include <cstdio>
#include <cuda_runtime.h>

/*struct int_pair
{
    int x;
    int y;
    constexpr int_pair(int x_, int y_) : x(x_), y(y_) { }
    constexpr int_pair() : int_pair(0,0) { }
};*/

struct neco
{
    dim3 p1;
    dim3 p2;
    constexpr neco(int x1, int y1, int x2, int y2) : p1(x1,y1), p2(x2,y2) { }
    template<int n> constexpr const dim3 & get() const {
        switch(n) {
            case 2:
                return p2;
            case 1:
            default:
                return p1;
        }
    }
};


int main()
{
    constexpr neco abc(11,12,21,22);

    constexpr int n = 2;

    constexpr int a = abc.get<n>().x;

    printf("%d\n", a);


    return 0;
}
