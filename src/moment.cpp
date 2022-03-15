#include "moment.hpp"

template<typename P>
moment<P>::moment(std::vector<vector_func<P>> md_func) : md_func_(md_func)
{
}

template class moment<float>;
template class moment<double>;
