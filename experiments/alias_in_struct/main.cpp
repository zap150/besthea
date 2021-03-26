#include <cstdio>

struct Man
{
    int age;
    float height;
};

struct Woman
{
    int age;
    float height;
    int age_2;
    float height_2;
};

struct Person
{
    int age;
    float height;
    int &how_old_he_is = age;
    float &how_high_is_he = height;
};

struct Alien
{
    int age;
    float height;
    int &how_old_he_is {age};
    float &how_high_is_he {height};
};



int main()
{

    Person x;
    x.age = 23;
    x.height = 'J';

    printf("sizeof(Man)    = %2lu\n", sizeof(Man));
    printf("sizeof(Woman)  = %2lu\n", sizeof(Woman));
    printf("sizeof(Person) = %2lu\n", sizeof(Person));
    printf("sizeof(Alien)  = %2lu\n", sizeof(Person));
    printf("how old he is: %d\n", x.how_old_he_is);
    printf("how is he called: %f\n", x.how_high_is_he);



    return 0;
}
