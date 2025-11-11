// test4.c
#include <stdio.h>

int square(int x) {
    return x * x;
}

int main() {
    int val = 6;
    int sq = square(val);
    printf("Square of %d is %d\n", val, sq);
    return 0;
}
