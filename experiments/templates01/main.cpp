#include <cstdio>


template<class T>
class Storage {
private:
    T value;
public:
    Storage(T val);
    T Get();
    void Print();
    void PrintFiveTimes();
};

template<class T>
Storage<T>::Storage(T val) {
    this->value = val;
}

template<class T>
T Storage<T>::Get() {
    return value;
}

template<>
void Storage<int>::Print() {
    printf("This is int of value %d\n", value);
}

template<class T>
void Storage<T>::Print() {
    printf("I dont know what that is\n");
}

template<class T>
void Storage<T>::PrintFiveTimes() {
    for(int i = 0; i < 5; i++) {
        this->Print();
    }
}



int main(int argc, char **argv) {

    Storage<int> s1(42);
    Storage<double> s2(3.14);

    s1.Print();
    s2.Print();

    s1.PrintFiveTimes();
    s2.PrintFiveTimes();



    return 0;
}
