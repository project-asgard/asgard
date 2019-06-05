std::pair<std::string, double> continuity_3_MB(int level, int degree)
{
    

    level -= 1;
    degree -= 2;

    double a = 0.019688000000079125 * pow(level, 0) + -6.714436380952768 * pow(level, 1) + 12.834880095238516 * pow(level, 2) + -7.373541904762054 * pow(level, 3) + 1.3975347619047795 * pow(level, 4);
    double b = 0.008724800000019716 * pow(level, 0) + -3.603043276190669 * pow(level, 1) + 6.894563885714479 * pow(level, 2) + -3.9607251809524473 * pow(level, 3) + 0.7510742857142927 * pow(level, 4);
    double c = 0.29121600000006415 * pow(level, 0) + -9.645553961905268 * pow(level, 1) + 18.43217399047674 * pow(level, 2) + -10.588897009523999 * pow(level, 3) + 2.007082923809546 * pow(level, 4);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 9bac2a8487482223d33f8606df17dc83ef6e79f1 on Tuesday, June 04 2019 at  1:01 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}