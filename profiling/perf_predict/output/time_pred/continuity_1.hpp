std::pair<std::string, double> continuity_1_seconds(int level, int degree)
{
    level -= 1;
    degree -= 2;

    double a = -0.0028571428571429703 * pow(level, 0) + -0.0007738095238086923 * pow(level, 1) + -0.0139841269841301 * pow(level, 2) + 0.015657738095240827 * pow(level, 3) + -0.004960317460318591 * pow(level, 4) + 0.0004761904761907268 * pow(level, 5) + 1.587301587298791e-05 * pow(level, 6) + -2.9761904761892527e-06 * pow(level, 7);
    double b = 0.00942857142856961 * pow(level, 0) + 0.08941190476190691 * pow(level, 1) + -0.07638571428570985 * pow(level, 2) + 0.010345436507928263 * pow(level, 3) + 0.0018273809523857377 * pow(level, 4) + 0.0002924603174590545 * pow(level, 5) + -0.00022738095238079666 * pow(level, 6) + 2.1626984126976837e-05 * pow(level, 7);
    double c = 0.11228571428558057 * pow(level, 0) + 0.1182809523809576 * pow(level, 1) + -0.4590404761903165 * pow(level, 2) + 0.5031293650790883 * pow(level, 3) + -0.24385119047602882 * pow(level, 4) + 0.05898293650789392 * pow(level, 5) + -0.00696547619047097 * pow(level, 6) + 0.00032103174603150306 * pow(level, 7);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 9bac2a8487482223d33f8606df17dc83ef6e79f1 on Tuesday, June 04 2019 at 12:56 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}