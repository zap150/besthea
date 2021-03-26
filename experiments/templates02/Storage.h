#pragma once

template<class T>
class Storage {
private:
    T value;
public:
    Storage(T initVal);
    T Get();
};
