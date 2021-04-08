#include <cstdio>


class Neco
{
private:
    int x;
public:
    Neco(int x_)
    {
        this->x = x_;
    }
    // void Print(int a, int b = x) // cannot be used like that
    // {
    //     printf("Printing a=%d b=%d\n", a, b);
    // }
    void Print(int a, int b)
    {
        printf("Printing a=%d b=%d\n", a, b);
    }
    void Print(int a)
    {
        Print(a, x);
    }
};


int main()
{
    Neco n(17);

    n.Print(3,5);
    n.Print(8);



    return 0;
}
