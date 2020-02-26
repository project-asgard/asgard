std::pair<std::string, double> continuity_2_seconds(int level, int degree)
{
    level -= 1;
    degree -= 2;

    double a = 0.0009523809523809575 * pow(level, 0) + 0.002718253968254221 * pow(level, 1) + -0.00752976190476221 * pow(level, 2) + 0.004305555555555671 * pow(level, 3) + -0.0006845238095238234 * pow(level, 4);
    double b = 0.0035238095238095167 * pow(level, 0) + -0.009567460317461193 * pow(level, 1) + 0.026216269841270887 * pow(level, 2) + -0.015027777777778169 * pow(level, 3) + 0.0028075396825397317 * pow(level, 4);
    double c = 0.10323809523809548 * pow(level, 0) + -0.028007936507937935 * pow(level, 1) + 0.05005158730158866 * pow(level, 2) + -0.028944444444444856 * pow(level, 3) + 0.006519841269841311 * pow(level, 4);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 8e150698e3c34ad032fe97a5a0475d276b726c18 on Tuesday, June 04 2019 at  2:41 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}