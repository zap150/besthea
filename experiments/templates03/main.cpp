#include <cstdio>


template<class T>
class Base
{
public:
    T value;
    Base(T val) {
        value = val;
    }
    T Get() {
        return value;
    }
};

template<class T>
class Something : public Base<T>
{
public:
    void Set(T val) {
        this->value = val; // "this" keyword is required if the classes are templated....
    }
};



int main()
{





}
