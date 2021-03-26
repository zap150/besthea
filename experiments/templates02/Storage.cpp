#include "Storage.h"



template<class T>
Storage<T>::Storage(T initVal) {
    value = initVal;
}


template<class T>
T Storage<T>::Get() {
    return value;
}



// THIS IS THE POINT
template class Storage<int>;
