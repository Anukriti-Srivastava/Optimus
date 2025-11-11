// test2.c
#include <stdio.h>

int main() {
    int x = 10;
    int y = 5;
    int max;

    if (x > y)
        max = x;
    else
        max = y;

    printf("Max is %d\n", max);
    return 0;
}
