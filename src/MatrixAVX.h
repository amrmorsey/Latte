//
// Created by shadyf on 07/09/17.
//

#ifndef INFERENCEENGINE_MATRIXAVX_H
#define INFERENCEENGINE_MATRIXAVX_H

#ifdef _WIN32
#include <malloc.h>
#endif
#include <avxintrin.h>
#include <xmmintrin.h>
#include <vector>
#include <cmath>

template <typename T, std::size_t Alignment>
class aligned_allocator
{
public:

    // The following will be the same for virtually all allocators.
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef ptrdiff_t difference_type;

    T * address(T& r) const
    {
        return &r;
    }

    const T * address(const T& s) const
    {
        return &s;
    }

    std::size_t max_size() const
    {
        // The following has been carefully written to be independent of
        // the definition of size_t and to avoid signed/unsigned warnings.
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }


    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, Alignment> other;
    } ;

    bool operator!=(const aligned_allocator& other) const
    {
        return !(*this == other);
    }

    void construct(T * const p, const T& t) const
    {
        void * const pv = static_cast<void *>(p);

        new (pv) T(t);
    }

    void destroy(T * const p) const
    {
        p->~T();
    }

    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const aligned_allocator& other) const
    {
        return true;
    }


    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    aligned_allocator() { }

    aligned_allocator(const aligned_allocator&) { }

    template <typename U> aligned_allocator(const aligned_allocator<U, Alignment>&) { }

    ~aligned_allocator() { }


    // The following will be different for each allocator.
    T * allocate(const std::size_t n) const
    {
        // The return value of allocate(0) is unspecified.
        // Mallocator returns NULL in order to avoid depending
        // on malloc(0)'s implementation-defined behavior
        // (the implementation can define malloc(0) to return NULL,
        // in which case the bad_alloc check below would fire).
        // All allocators can return NULL in this case.
        if (n == 0) {
            return NULL;
        }

        // All allocators should contain an integer overflow check.
        // The Standardization Committee recommends that std::length_error
        // be thrown in the case of integer overflow.
        if (n > max_size())
        {
            throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void * const pv = _mm_malloc(n * sizeof(T), Alignment);

        // Allocators should throw std::bad_alloc in the case of memory allocation failure.
        if (pv == NULL)
        {
            throw std::bad_alloc();
        }

        return static_cast<T *>(pv);
    }

    void deallocate(T * const p, const std::size_t n) const
    {
        _mm_free(p);
    }


    // The following will be the same for all allocators that ignore hints.
    template <typename U>
    T * allocate(const std::size_t n, const U * /* const hint */) const
    {
        return allocate(n);
    }


    // Allocators are not required to be assignable, so
    // all allocators should have a private unimplemented
    // assignment operator. Note that this will trigger the
    // off-by-default (enabled under /Wall) warning C4626
    // "assignment operator could not be generated because a
    // base class assignment operator is inaccessible" within
    // the STL headers, but that warning is useless.
private:
    aligned_allocator& operator=(const aligned_allocator&);
};

typedef std::vector<__m256, aligned_allocator<__m256, sizeof(__m256)> > aligned_vector;

class MatrixAVX {
private:

//    https://stackoverflow.com/questions/13879609/horizontal-sum-of-8-packed-32bit-floats
    static inline __m256 hsums(__m256 const &v) {
        auto x = _mm256_permute2f128_ps(v, v, 1);
        auto y = _mm256_add_ps(v, x);
        x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
        x = _mm256_add_ps(x, y);
        y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_ps(x, y);
    }

    aligned_vector xmm;
    unsigned long matrix_size;
    std::vector<int> matrix_shape;
    unsigned long xmm_size;
public:

    explicit MatrixAVX(std::vector<float> vec, std::vector<int> shape) : matrix_shape(shape){
        matrix_size = 1;

        for (int x : shape)
            matrix_size *= x;

        xmm_size = static_cast<unsigned long>(ceil(matrix_size / 8.0f));

        unsigned long aligned_size = matrix_size / matrix_size % 8;

        for (int i = 0; i < aligned_size; i++) {
            xmm.push_back(_mm256_loadu_ps(&vec[i * 8]));
        }

        // Check for stranglers in case matrix size is not divisible by 8
        int rem = static_cast<int>(matrix_size % 8);
        if(rem) {
            // Set up a mask for partial loading.
            // Highest bit -> 1 in mask element means corresponding array element will be taken (i.e negative values)
            // Highest bit -> 0 in mask element means corresponding array element will be taken as 0
            __m256i mask = _mm256_setr_epi32(-rem, 1-rem, 2-rem, 3-rem, 4-rem, 5-rem, 6-rem, 7-rem);
            xmm.push_back(_mm256_maskload_ps(&vec[aligned_size*8], mask));
        }
    };

    static void dot_product(MatrixAVX const &a, MatrixAVX &out) {
//        __m256 c = _mm256_mul_ps(evens, odds);
    }

};


#endif //INFERENCEENGINE_MATRIXAVX_H
