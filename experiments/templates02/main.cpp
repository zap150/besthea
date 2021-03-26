#include <cstdio>
#include "Storage.h"

int main() {

    Storage<int> s(42);

    printf("Stored value is %d\n", s.Get());

    return 0;
}

