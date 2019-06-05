std::pair<std::string, double> continuity_2_MB(int level, int degree)
{
    

    level -= 1;
    degree -= 2;

    double a = 0.0018040000000002338 * pow(level, 0) + -0.040294047619050054 * pow(level, 1) + 0.07590307142857423 * pow(level, 2) + -0.04248966666666777 * pow(level, 3) + 0.008369214285714426 * pow(level, 4);
    double b = -0.003178400000000116 * pow(level, 0) + -0.12443880952381585 * pow(level, 1) + 0.2719346142857203 * pow(level, 2) + -0.15975953333333504 * pow(level, 3) + 0.03236944285714301 * pow(level, 4);
    double c = 0.27665440000000424 * pow(level, 0) + -0.3649787619047832 * pow(level, 1) + 0.6924633428571662 * pow(level, 2) + -0.39200306666667484 * pow(level, 3) + 0.07736442857142951 * pow(level, 4);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 8e150698e3c34ad032fe97a5a0475d276b726c18 on Tuesday, June 04 2019 at  2:39 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}