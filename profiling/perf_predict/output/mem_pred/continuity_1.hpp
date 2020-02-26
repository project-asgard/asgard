std::pair<std::string, double> continuity_1_MB(int level, int degree)
{
    

    level -= 1;
    degree -= 2;

    double a = 0.0006674285713964426 * pow(level, 0) + 0.08929173605442808 * pow(level, 1) + -0.21685896666659416 * pow(level, 2) + 0.2039998134919392 * pow(level, 3) + -0.0956067857142133 * pow(level, 4) + 0.02410198730156825 * pow(level, 5) + -0.0031291047619024293 * pow(level, 6) + 0.00016932029478447202 * pow(level, 7);
    double b = -0.0017353142858960336 * pow(level, 0) + 0.36947940625850806 * pow(level, 1) + -0.940917886666319 * pow(level, 2) + 0.908441653808946 * pow(level, 3) + -0.43068029047585715 * pow(level, 4) + 0.10871560190467436 * pow(level, 5) + -0.014034394285703564 * pow(level, 6) + 0.0007495094557818137 * pow(level, 7);
    double c = 0.27672845714285105 * pow(level, 0) + 0.4088480397279247 * pow(level, 1) + -1.0097636355552568 * pow(level, 2) + 0.9382915292058192 * pow(level, 3) + -0.4330815603171498 * pow(level, 4) + 0.10748451015864818 * pow(level, 5) + -0.013742118412688364 * pow(level, 6) + 0.0007312351927432963 * pow(level, 7);

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        "(Predicted for 9bac2a8487482223d33f8606df17dc83ef6e79f1 on Tuesday, June 04 2019 at 12:56 pm)",
        a * pow(degree, 2) + b * degree + c
    );
}