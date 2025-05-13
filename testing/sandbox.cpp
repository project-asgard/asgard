#include "asgard.hpp"

// using namespace asgard;

using P = asgard::default_precision;

using term_1d = asgard::term_1d<P>;

using term_md = asgard::term_md<P>;

using term_div = asgard::term_div<P>;

using term_grad = asgard::term_grad<P>;

using term_volume = asgard::term_volume<P>;

using separable_func = asgard::separable_func<P>;

P sgn(P x){
    return (x >= 0) ? 1 : -1;
}

P A_func(P){
   return 1;
}

template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_pde(asgard::prog_opts options) {

    P const x_min = -1;

    P const x_max = 1;

    asgard::pde_domain<P> domain(std::vector<asgard::domain_range>(1, {x_min,x_max}));

    options.default_degree = 2;

    options.default_start_levels = {10, };

    options.force_step_method(asgard::time_method::steady);

    options.default_solver = asgard::solver_method::direct;

    options.default_isolver_tolerance= 1.E-8;

    options.default_isolver_iterations = 1000;


    asgard::pde_scheme<P> pde(options, std::move(domain));

    P const dx = pde.min_cell_size();

    { // adding the source component
        auto source_vec = [](const std::vector<P> &x, P /* time */,
                             std::vector<P> &func) {
            for (size_t i = 0; i < x.size(); i++)
                func[i] = 1;
        };

        separable_func src({source_vec, }, asgard::ignores_time);
        // if the source will be a constant, it is better to set it as such
        // for example, src.set(0, P{1}); where 0 is the dimension

        pde.add_source(src);
    }

    { // C*u
        // if you want to do testing of your code, do not set C = 0
        // instead, comment out the line that says "pde += u"
        auto C = [](std::vector<P> const &x, std::vector<P> &func) ->
        void {
            for (size_t i = 0; i < x.size(); i++)
                func[i] = 0;
        };
        term_1d u = term_volume{C};

        pde += u;
    }

    { // B*d/dv (g*u)
        auto B = [](std::vector<P> const &x, std::vector<P> &func) ->
        void {
            for (size_t i = 0; i < x.size(); i++)
                func[i] = 1;
        };

        auto g = [](std::vector<P> const &x, std::vector<P> &func) ->
        void {
            for (size_t i = 0; i < x.size(); i++)
                func[i] = -1;
        };

        term_grad div1(g, asgard::flux_type::upwind);

        term_1d u_x({term_volume{B},div1});

        u_x.set_penalty((x_max-x_min) / dx);

        pde += u_x;
    }

    { // A*d/dv (f*du/dv)
        auto f = [](std::vector<P> const &x, std::vector<P> &func) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            func[i] = 2;
        };

        auto A = [](std::vector<P> const &x, std::vector<P> &func) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            func[i] = 3;
        };

        term_div div2(f, asgard::flux_type::upwind);

        term_grad grad(-1, asgard::flux_type::upwind, asgard::boundary_type::bothsides);

        term_1d u_xx({term_volume{A}, div2, term_volume{f}, grad});

        u_xx.set_penalty((x_max-x_min)/ dx);

        term_md u_xx_md({u_xx, });

        separable_func bc(asgard::ones_for_dimensions{1}, asgard::ignores_time);

        bc.set(0, 1.0); // ones_for_dimensions{1} already set this to 1

        asgard::boundary_flux<P> rbf = asgard::right_boundary_flux(bc);

        u_xx_md += rbf;

        pde+=u_xx_md;
    }

    // A*d/dv (f*du/dv)+B*d/dv (g*u)+C*u = S
    return pde;
}

int main(int argc, char** argv)
{
    // parse the command-line inputs
    asgard::prog_opts options(argc, argv);

    asgard::discretization_manager<P> disc(make_pde(options),
                                           asgard::verbosity_level::high);

    disc.advance_time();

    disc.final_output();

    return 0;
}
