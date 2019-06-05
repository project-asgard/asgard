std::pair<std::string, double> continuity_3_seconds(int level, int degree)
{
    level -= 1;
    degree -= 2;

    double a = 0.002857142857144573 * pow(level, 0) + -0.38630952380954076 * pow(level, 1) + 0.6724404761904947 * pow(level, 2) + -0.33583333333333976 * pow(level, 3) + 0.05184523809523887 * pow(level, 4);
    double b = -0.0014285714285782445 * pow(level, 0) + 1.0062380952381387 * pow(level, 1) + -1.7052619047619495 * pow(level, 2) + 0.7983333333333495 * pow(level, 3) + -0.09888095238095432 * pow(level, 4);
    double c = 0.10971428571429791 * pow(level, 0) + -1.2576190476191125 * pow(level, 1) + 2.3733809523810225 * pow(level, 2) + -1.3166666666666913 * pow(level, 3) + 0.23319047619047908 * pow(level, 4);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 9bac2a8487482223d33f8606df17dc83ef6e79f1 on Tuesday, June 04 2019 at  1:01 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}