//
// This would otherwise be the start of the tensors.cpp, if we were still doing
// the explicit instantiations
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace asgard
{
template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename>
inline void fk::copy_vector(fk::vector<P, mem, resrc> &dest,
                            fk::vector<P, omem, oresrc> const &source)
{
  expect(source.size() == dest.size());
  fk::memcpy_1d<resrc, oresrc>(dest.data(), source.data(), source.size());
}

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector() : data_{nullptr}, size_{0}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(int const size) : size_{size}
{
  expect(size >= 0);
  allocate_resource<resrc>(data_, size_);
}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(std::initializer_list<P> list)
    : size_{static_cast<int>(list.size())}
{
  allocate_resource<resrc>(data_, size_, false);
  memcpy_1d<resrc, resource::host>(data_, list.begin(), size_);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem, resrc>::vector(std::vector<P> const &v)
    : data_{new P[v.size()]}, size_{static_cast<int>(v.size())}
{
  std::copy(v.begin(), v.end(), data_);
}

//
// matrix conversion constructor linearizes the matrix, i.e. stacks the columns
// of the matrix into a single vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(
    fk::matrix<P, mem_type::owner, resrc> const &mat)
    : data_(nullptr), size_(mat.size())
{
  if (size_ != 0)
  {
    allocate_resource<resrc>(data_, size_, false);
    memcpy_1d<resrc, resrc>(data_, mat.data(), mat.size());
  }
}

// vector view constructor given a start and stop index
// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// const view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// delegating constructor to extract view from owner. overload for default case
// of viewing the entire owner
// const view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// modifiable view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// create vector view of an existing matrix
// const version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, 0, column_index, row_start, row_stop)
{}

// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, 0, column_index, row_start, row_stop)
{}

template<typename P, mem_type mem, resource resrc>
#ifdef __clang__
fk::vector<P, mem, resrc>::~vector<P, mem, resrc>()
#else
fk::vector<P, mem, resrc>::~vector()
#endif
{
  if constexpr (mem == mem_type::owner)
  {
    delete_resource<resrc>(data_);
  }
}

//
// vector copy constructor for like types (like types only)
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> const &a)
    : size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    allocate_resource<resrc>(data_, a.size(), false);
    copy_vector(*this, a);
  }
  else
  {
    // working with view, OK to alias
    data_ = const_cast<P *>(a.data_);
  }
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> const &a)
{
  static_assert(mem != mem_type::const_view,
                "cannot copy assign into const_view!");

  if (&a == this)
    return *this;

  expect(size() == a.size());

  copy_vector(*this, a);

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> &&a)
    : data_{a.data_}, size_{a.size_}
{
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> &&a)
{
  static_assert(mem != mem_type::const_view,
                "cannot move assign into const_view!");

  if (&a == this)
    return *this;

  size_ = a.size_;
  P *const temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

// copy construct owner from view values
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, mem_type, typename>
fk::vector<P, mem, resrc>::vector(vector<P, omem, resrc> const &a)
    : size_(a.size())
{
  allocate_resource<resrc>(data_, a.size(), false);
  copy_vector(*this, a);
}

// assignment owner <-> view
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, omem, resrc> const &a)
{
  expect(size() == a.size());
  copy_vector(*this, a);
  return *this;
}

// transfer functions
// host->dev, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::device>
fk::vector<P, mem, resrc>::clone_onto_device() const

{
  fk::vector<P, mem_type::owner, resource::device> a(size());
  copy_vector(a, *this);
  return a;
}

// host->dev copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    fk::vector<P, omem, resource::host> const &a)
{
  expect(a.size() == size());
  copy_vector(*this, a);
  return *this;
}

// dev -> host, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::host>
fk::vector<P, mem, resrc>::clone_onto_host() const

{
  fk::vector<P> a(size());
  copy_vector(a, *this);
  return a;
}

// dev -> host copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    vector<P, omem, resource::device> const &a)
{
  expect(a.size() == size());
  copy_vector(*this, a);
  return *this;
}

//
// copy out of std::vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::operator=(std::vector<P> const &v)
{
  expect(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(P));
  return *this;
}

//
// copy into std::vector
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
std::vector<P> fk::vector<P, mem, resrc>::to_std() const
{
  return std::vector<P>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::vector<P, mem, resrc>::operator()(int i)
{
  expect(i < size_);
  return data_[i];
}

template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P const &fk::vector<P, mem, resrc>::operator()(int i) const
{
  expect(i < size_);
  return data_[i];
}

// array index operators
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::vector<P, mem, resrc>::operator[](int i)
{
  expect(i < size_);
  return data_[i];
}
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P const &fk::vector<P, mem, resrc>::operator[](int i) const
{
  expect(i < size_);
  return data_[i];
}

// vector comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME do we need to be more careful with these fp comparisons?
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator==(vector<P, omem> const &other) const
{
  if constexpr (omem == mem)
    if (&other == this)
      return true;
  if (size() != other.size())
    return false;
  for (auto i = 0; i < size(); ++i)
    if constexpr (std::is_floating_point_v<P>)
    {
      if (std::abs((*this)(i)-other(i)) > TOL)
      {
        return false;
      }
    }
    else
    {
      if ((*this)(i) != other(i))
      {
        return false;
      }
    }
  return true;
}
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator!=(vector<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator<(vector<P, omem> const &other) const
{
  return std::lexicographical_compare(begin(), end(), other.begin(),
                                      other.end());
}

//
// vector addition operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator+(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator-(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem>
P fk::vector<P, mem, resrc>::operator*(
    vector<P, omem, resrc> const &right) const
{
  expect(size() == right.size());
  vector const &X = (*this);
  return lib_dispatch::dot<resrc>(size(), X.data(), 1, right.data(), 1);
}

//
// vector*matrix multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator*(fk::matrix<P, omem, resrc> const &A) const
{
  // check dimension compatibility
  expect(size() == A.nrows());

  vector const &X = (*this);
  vector<P> Y(A.ncols());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P zero = 0.0;
  P one  = 1.0;
  lib_dispatch::gemv<resrc>('t', m, n, one, A.data(), lda, X.data(), one_i,
                            zero, Y.data(), one_i);
  return Y;
}

//
// vector*scalar multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::operator*(P const x) const
{
  vector<P> a(*this);
  int one_i = 1;
  int n     = a.size();
  P alpha   = x;

  lib_dispatch::scal(n, alpha, a.data(), one_i);

  return a;
}

//
// perform the matrix kronecker product by
// interpreting vector operands/return vector
// as single column matrices.
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::single_column_kron(
    vector<P, omem> const &right) const
{
  fk::vector<P> product((*this).size() * right.size());
  for (int j = 0; j < right.size(); ++j)
  {
    for (int i = 0; i < (*this).size(); ++i)
    {
      product(i * right.size() + j) = (*this)(i)*right(j);
    }
  }
  return product;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::scale(P const x)
{
  int one_i = 1;
  int n     = this->size();
  P alpha   = x;

  lib_dispatch::scal<resrc>(n, alpha, this->data(), one_i);

  return *this;
}

//
// utility functions
//

//
// Prints out the values of a vector
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::print(std::string_view const label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner)" << '\n';
  else if constexpr (mem == mem_type::view)
    std::cout << label << "(view)" << '\n';
  else if constexpr (mem == mem_type::const_view)
    std::cout << label << "(const view)" << '\n';
  else
    expect(false); // above cases should cover all implemented types

  if constexpr (std::is_floating_point_v<P>)
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                << std::right << (*this)(i);
  }
  else
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::right << (*this)(i) << " ";
  }
  std::cout << '\n';
}

//
// Dumps to file a vector that can be read straight into octave
// Same as the matrix:: version
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::dump_to_octave(
    std::filesystem::path const &filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < size(); ++i)
    std::cout << std::setprecision(12) << (*this)(i) << " ";

  std::cout.rdbuf(coutbuf);
}

//
// resize the vector
// (currently supports a subset of the std::vector.resize() interface)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem_type::owner, resrc> &
fk::vector<P, mem, resrc>::resize(int const new_size)
{
  expect(new_size >= 0);
  if (new_size == this->size())
    return *this;
  P *old_data{data_};

  if constexpr (resrc == resource::host)
  {
    data_ = new P[new_size];
    if (size() > 0 && new_size > 0)
    {
      if (size() < new_size)
      {
        std::memcpy(data_, old_data, size() * sizeof(P));
        std::fill(data_ + size(), data_ + new_size, P{0});
      }
      else
        std::memcpy(data_, old_data, new_size * sizeof(P));
    }
    delete[] old_data;
  }
  else
  {
    allocate_device(data_, new_size);
    if (size() > 0 && new_size > 0)
    {
      if (size() < new_size)
        copy_on_device(data_, old_data, size());
      else
        copy_on_device(data_, old_data, new_size);
    }
    delete_device(old_data);
  }
  size_ = new_size;
  return *this;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::concat(vector<P, omem> const &right)
{
  int const old_size = this->size();
  int const new_size = this->size() + right.size();
  P *old_data{data_};
  data_ = new P[new_size];
  std::memcpy(data_, old_data, old_size * sizeof(P));
  std::memcpy(data(old_size), right.data(), right.size() * sizeof(P));
  size_ = new_size;
  delete[] old_data;
  return *this;
}

// set a subvector beginning at provided index
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::set_subvector(int const index,
                                         fk::vector<P, omem> const sub_vector)
{
  expect(index >= 0);
  expect((index + sub_vector.size()) <= this->size());
  std::memcpy(&(*this)(index), sub_vector.data(),
              sub_vector.size() * sizeof(P));
  return *this;
}

// extract subvector, indices inclusive
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem_type::owner, resrc>
fk::vector<P, mem, resrc>::extract(int const start, int const stop) const
{
  expect(start >= 0);
  expect(stop < this->size());
  expect(stop >= start);

  int const sub_size = stop - start + 1;
  fk::vector<P, mem_type::owner, resrc> sub_vector(sub_size);
  lib_dispatch::copy<resrc>(sub_size, data(start), sub_vector.data());
  return sub_vector;
}

// const/nonconst view constructors delegate to this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index,
                                  bool const delegated)
{
  // ignore dummy argument to avoid compiler warnings
  ignore(delegated);
  data_ = nullptr;
  size_ = 0;
  if (vec.size() > 0)
  {
    expect(start_index >= 0);
    expect(stop_index < vec.size());
    expect(stop_index >= start_index);

    data_ = vec.data_ + start_index;
    size_ = stop_index - start_index + 1;
  }
}

// public const/nonconst vector view from matrix constructors delegate to
// this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source, int,
                                  int const column_index, int const row_start,
                                  int const row_stop)
{
  expect(column_index >= 0);
  expect(column_index < source.ncols());
  expect(row_start >= 0);
  expect(row_start <= row_stop);
  expect(row_stop < source.nrows());

  data_ = nullptr;
  size_ = row_stop - row_start + 1;

  if (size_ > 0)
  {
    data_ = const_cast<P *>(source.data(
        int64_t{column_index} * source.stride() + int64_t{row_start}));
  }
}
} // namespace asgard
