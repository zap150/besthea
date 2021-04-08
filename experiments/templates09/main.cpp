#include <cstdio>


template<typename T>
class Container
{
private:
    T value;
public:
    Container(T value);
    const T& Get() const;
    void Print() const;

    template<int count>
    void PrintMany() const;
};

template<typename T>
Container<T>::Container(T value_)
{
    this->value = value_;
}

template<typename T>
const T& Container<T>::Get() const
{
    return value;
}

template<typename T>
void Container<T>::Print() const
{
    int * val = (int*)(&value);
    printf("%x\n", *val);
}

template<typename T>
template<int count>
void Container<T>::PrintMany() const
{
    for(int i = 0; i < count; i++)
        Print();
}


int main()
{
    Container<int> c(42);

    c.PrintMany<3>();


    return 0;
}
