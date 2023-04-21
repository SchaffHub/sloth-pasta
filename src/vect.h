/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __PASTA_VDF_ASM_VECT_H__
#define __PASTA_VDF_ASM_VECT_H__

#include <stddef.h>

#if defined(__x86_64__) || defined(__aarch64__)
/* These are available even in ILP32 flavours, but even then they are
 * capable of performing 64-bit operations as efficiently as in *P64. */
typedef unsigned long long limb_t;
# define LIMB_T_BITS    64

#elif defined(_WIN64)   /* Win64 is P64 */
typedef unsigned __int64 limb_t;
# define LIMB_T_BITS    64

#else
# error "unsupported platform"
#endif

#define TO_LIMB_T(limb64)     limb64

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

typedef limb_t vec256[NLIMBS(256)];
typedef limb_t vec512[NLIMBS(512)];

/*
 * Internal Boolean type, Bolean by value, hence safe to cast to or
 * reinterpret as 'bool'.
 */
typedef limb_t bool_t;

/*
 * Assembly subroutines...
 */
#if defined(__ADX__) /* e.g. -march=broadwell */ && !defined(__PASTA_PORTABLE__)
# define mul_mont_pasta mulx_mont_pasta
# define sqr_mont_pasta sqrx_mont_pasta
# define from_mont_pasta fromx_mont_pasta
# define redc_mont_pasta redcx_mont_pasta
#endif

extern "C"
void mul_mont_pasta(vec256 ret, const vec256 a, const vec256 b,
                    const vec256 p, limb_t n0);
extern "C"
void sqr_mont_pasta(vec256 ret, const vec256 a, const vec256 p, limb_t n0);
extern "C"
void redc_mont_pasta(vec256 ret, const vec512 a, const vec256 p, limb_t n0);
extern "C"
void from_mont_pasta(vec256 ret, const vec256 a, const vec256 p, limb_t n0);

extern "C"
void add_mod_pasta(vec256 ret, const vec256 a, const vec256 b, const vec256 p);
extern "C"
void sub_mod_pasta(vec256 ret, const vec256 a, const vec256 b, const vec256 p);

#ifdef __UINTPTR_TYPE__
typedef __UINTPTR_TYPE__ uptr_t;
#else
typedef const void *uptr_t;
#endif

#if !defined(restrict)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define restrict __restrict__
#  elif defined(_MSC_VER)
#   define restrict __restrict
#  else
#   define restrict
#  endif
# endif
#endif

#if defined(__CUDA_ARCH__)
# define inline inline __device__
#endif

#if !defined(inline) && !defined(__cplusplus)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define inline __inline__
#  elif defined(_MSC_VER)
#   define inline __inline
#  else
#   define inline
#  endif
# endif
#endif

static inline bool_t is_zero(limb_t l)
{   return (~l & (l - 1)) >> (LIMB_T_BITS - 1);   }

static inline bool_t vec_is_zero(const void *a, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    limb_t acc;
    size_t i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i];

    return is_zero(acc);
}

static inline bool_t vec_is_equal(const void *a, const void *b, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    const limb_t *bp = (const limb_t *)b;
    limb_t acc;
    size_t i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i] ^ bp[i];

    return is_zero(acc);
}

static inline void vec_copy(void *restrict ret, const void *a, size_t num)
{
    limb_t *rp = (limb_t *)ret;
    const limb_t *ap = (const limb_t *)a;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
}

static inline void vec_zero(void *ret, size_t num)
{
    volatile limb_t *rp = (volatile limb_t *)ret;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = 0;

#if defined(__GNUC__) && !defined(__NVCC__)
    asm volatile("" : : "r"(ret) : "memory");
#endif
}

static inline void limbs_from_be_bytes(limb_t *restrict ret,
                                       const unsigned char *in, size_t n)
{
    limb_t limb = 0;

    while(n--) {
        limb <<= 8;
        limb |= *in++;
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restict'-ed storage...
         */
        ret[n / sizeof(limb_t)] = limb;
    }
}

static inline void be_bytes_from_limbs(unsigned char *out, const limb_t *in,
                                       size_t n)
{
    limb_t limb;

    while(n--) {
        limb = in[n / sizeof(limb_t)];
        *out++ = (unsigned char)(limb >> (8 * (n % sizeof(limb_t))));
    }
}

static inline void limbs_from_le_bytes(limb_t *restrict ret,
                                       const unsigned char *in, size_t n)
{
    limb_t limb = 0;

    while(n--) {
        limb <<= 8;
        limb |= in[n];
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restict'-ed storage...
         */
        ret[n / sizeof(limb_t)] = limb;
    }
}

static inline void le_bytes_from_limbs(unsigned char *out, const limb_t *in,
                                       size_t n)
{
    const union {
        long one;
        char little;
    } is_endian = { 1 };
    limb_t limb;
    size_t i, j, r;

    if ((uptr_t)out == (uptr_t)in && is_endian.little)
        return;

    r = n % sizeof(limb_t);
    n /= sizeof(limb_t);

    for(i = 0; i < n; i++) {
        for (limb = in[i], j = 0; j < sizeof(limb_t); j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
    if (r) {
        for (limb = in[i], j = 0; j < r; j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
}

/*
 * Some compilers get arguably overzealous(*) when passing pointer to
 * multi-dimensional array [such as vec384x] as 'const' argument.
 * General direction seems to be to legitimize such constification,
 * so it's argued that suppressing the warning is appropriate.
 *
 * (*)  http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1923.htm
 */
#if defined(__INTEL_COMPILER)
# pragma warning(disable:167)
# pragma warning(disable:556)
#elif defined(__GNUC__) && !defined(__clang__)
# pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(_MSC_VER)
# pragma warning(disable: 4127 4189)
#endif

#include <stdlib.h>

#if defined(__GNUC__)
# ifndef alloca
#  define alloca(s) __builtin_alloca(s)
# endif
#elif defined(__sun)
# include <alloca.h>
#elif defined(_WIN32)
# include <malloc.h>
# ifndef alloca
#  define alloca(s) _alloca(s)
# endif
#endif

#endif /* __PASTA_VDF_ASM_VECT_H__ */
