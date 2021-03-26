#include "header.h"

template<class T>
void Neco<T>::Set(T val)
{
    this->value = val;
}


template class Neco<int>; // it has to be in this file too
