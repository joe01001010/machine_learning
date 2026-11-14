#include <stdio.h>

int main(void) {
    double base, per_minute, per_mile, miles, minimum;
    int minutes;
    if (scanf("%lf %lf %lf %lf %d %lf",
              &base, &per_minute, &per_mile, &miles, &minutes, &minimum) != 6) {
        return 1;
    }
    double subtotal = base + per_minute * minutes + per_mile * miles;
    printf("%.2f\n", subtotal < minimum ? minimum : subtotal);
    return 0;
}
