#pragma once


template<class T>
class Neco
{
private:
    T value;
public:
    Neco(T val);
    T Get();
    void Set(T val);
};
