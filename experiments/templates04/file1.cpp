#include "header.h"

template<class T>
Neco<T>::Neco(T val)
{
    this->value = val;
}

template<class T>
T Neco<T>::Get()
{
    return this->value;
}


template class Neco<int>;
