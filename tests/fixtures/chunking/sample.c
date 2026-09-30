#include <stdio.h>

#define MAX 8

struct point {
    int x;
    int y;
};

typedef struct {
    int size;
} buffer;

static int *find_slot(int *slots, int n) {
    return slots + n;
}

int main(void) {
    printf("%d\n", MAX);
    return 0;
}
