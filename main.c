#include <stdio.h>
#include <malloc.h>
#include <math.h>
#define N 100000000

double arr[10000001];

int main() {
    for(int i = 0; i<=10000000;i++)
       arr[i] = sin(2*M_PI*i/10000000);
    double sum=0;
    for(int i = 0;i<=10000000;i++)
        sum+=arr[i];
    printf("%le", sum);
    return 0;
}
