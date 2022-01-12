// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <immintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>
#include <xnnpack/unaligned.h>

void xnn_x32_transposec_ukernel__8x8_reuse_multi_avx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x32_transpose_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(float));
  assert(input_stride >= block_width * sizeof(float));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(float);
  const size_t tile_wbytes = tile_width * sizeof(float);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(float);

  const float* i0 = (const float*) input;
  float* o0 = (float*) output;

  do {
    float* o1 = (float*) (block_width < 2 ? o0 : (float*) ((uintptr_t) o0 + output_stride));
    float* o2 = (float*) (block_width <= 2 ? o0 : (float*) ((uintptr_t) o1 + output_stride));
    float* o3 = (float*) (block_width < 4 ? o0 : (float*) ((uintptr_t) o2 + output_stride));
    float* o4 = (float*) (block_width <= 4 ? o0 : (float*) ((uintptr_t) o3 + output_stride));
    float* o5 = (float*) (block_width < 6 ? o0 : (float*) ((uintptr_t) o4 + output_stride));
    float* o6 = (float*) (block_width <= 6 ? o0 : (float*) ((uintptr_t) o5 + output_stride));
    float* o7 = (float*) (block_width < 8 ? o0 : (float*) ((uintptr_t) o6 + output_stride));
    const size_t rem = min(block_width - 1, 7);

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7 - rem]));

    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m256 v3_0 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_1 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_2 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_3 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_4 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_5 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_6 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);
      const __m256 v3_7 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_stride);

      const __m256 v2_0 =  _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256 v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256 v2_2 =  _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256 v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256 v2_4 =  _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256 v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256 v2_6 =  _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256 v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256 v1_0 =  _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256 v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256 v1_2 =  _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256 v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256 v1_4 =  _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256 v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256 v1_6 =  _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256 v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);

      const __m256 v0_0 = _mm256_permute2f128_ps(v1_0, v1_4, 0x20);
      const __m256 v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
      const __m256 v0_1 = _mm256_permute2f128_ps(v1_1, v1_5, 0x20);
      const __m256 v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
      const __m256 v0_2 = _mm256_permute2f128_ps(v1_2, v1_6, 0x20);
      const __m256 v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
      const __m256 v0_3 = _mm256_permute2f128_ps(v1_3, v1_7, 0x20);
      const __m256 v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);

      _mm256_storeu_ps(o7, v0_7);
      o7 = (float*) ((uintptr_t) o7 + tile_hbytes);
      _mm256_storeu_ps(o6, v0_6);
      o6 = (float*) ((uintptr_t) o6 + tile_hbytes);
      _mm256_storeu_ps(o5, v0_5);
      o5 = (float*) ((uintptr_t) o5 + tile_hbytes);
      _mm256_storeu_ps(o4, v0_4);
      o4 = (float*) ((uintptr_t) o4 + tile_hbytes);
      _mm256_storeu_ps(o3, v0_3);
      o3 = (float*) ((uintptr_t) o3 + tile_hbytes);
      _mm256_storeu_ps(o2, v0_2);
      o2 = (float*) ((uintptr_t) o2 + tile_hbytes);
      _mm256_storeu_ps(o1, v0_1);
      o1 = (float*) ((uintptr_t) o1 + tile_hbytes);
      _mm256_storeu_ps(o0, v0_0);
      o0 = (float*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const __m256 v3_0 = _mm256_maskload_ps(i0, vmask);
      const float *i1 = (const float*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256 v3_1 = _mm256_maskload_ps(i1, vmask);
      const float *i2 = (const float*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256 v3_2 = _mm256_maskload_ps(i2, vmask);
      const float *i3 = (const float*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256 v3_3 = _mm256_maskload_ps(i3, vmask);
      const float *i4 = (const float*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256 v3_4 = _mm256_maskload_ps(i4, vmask);
      const float *i5 = (const float*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256 v3_5 = _mm256_maskload_ps(i5, vmask);
      const float *i6 = (const float*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256 v3_6 = _mm256_maskload_ps(i6, vmask);
      const __m256 v3_7 = _mm256_undefined_ps();

      const __m256 v2_0 =  _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256 v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256 v2_2 =  _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256 v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256 v2_4 =  _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256 v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256 v2_6 =  _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256 v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256 v1_0 =  _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256 v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256 v1_2 =  _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256 v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256 v1_4 =  _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256 v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256 v1_6 =  _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256 v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);

      __m256 v0_0 = _mm256_permute2f128_ps(v1_0, v1_4, 0x20);
      __m256 v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
      __m256 v0_1 = _mm256_permute2f128_ps(v1_1, v1_5, 0x20);
      __m256 v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
      __m256 v0_2 = _mm256_permute2f128_ps(v1_2, v1_6, 0x20);
      __m256 v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
      __m256 v0_3 = _mm256_permute2f128_ps(v1_3, v1_7, 0x20);
      __m256 v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);

      if (bh & 4) {
        _mm_storeu_ps(o7, _mm256_castps256_ps128(v0_7));
        o7 += 4;
        _mm_storeu_ps(o6, _mm256_castps256_ps128(v0_6));
        o6 += 4;
        _mm_storeu_ps(o5, _mm256_castps256_ps128(v0_5));
        o5 += 4;
        _mm_storeu_ps(o4, _mm256_castps256_ps128(v0_4));
        o4 += 4;
        _mm_storeu_ps(o3, _mm256_castps256_ps128(v0_3));
        o3 += 4;
        _mm_storeu_ps(o2, _mm256_castps256_ps128(v0_2));
        o2 += 4;
        _mm_storeu_ps(o1, _mm256_castps256_ps128(v0_1));
        o1 += 4;
        _mm_storeu_ps(o0, _mm256_castps256_ps128(v0_0));
        o0 += 4;
        v0_0 = _mm256_permute2f128_ps(v0_0, v0_0, 0x1);
        v0_1 = _mm256_permute2f128_ps(v0_1, v0_1, 0x1);
        v0_2 = _mm256_permute2f128_ps(v0_2, v0_2, 0x1);
        v0_3 = _mm256_permute2f128_ps(v0_3, v0_3, 0x1);
        v0_4 = _mm256_permute2f128_ps(v0_4, v0_4, 0x1);
        v0_5 = _mm256_permute2f128_ps(v0_5, v0_5, 0x1);
        v0_6 = _mm256_permute2f128_ps(v0_6, v0_6, 0x1);
        v0_7 = _mm256_permute2f128_ps(v0_7, v0_7, 0x1);
      }

      if (bh & 2) {
        _mm_storel_pd((double*) o7, _mm_castps_pd(_mm256_castps256_ps128(v0_7)));
        o7 += 2;
        _mm_storel_pd((double*) o6, _mm_castps_pd(_mm256_castps256_ps128(v0_6)));
        o6 += 2;
        _mm_storel_pd((double*) o5, _mm_castps_pd(_mm256_castps256_ps128(v0_5)));
        o5 += 2;
        _mm_storel_pd((double*) o4, _mm_castps_pd(_mm256_castps256_ps128(v0_4)));
        o4 += 2;
        _mm_storel_pd((double*) o3, _mm_castps_pd(_mm256_castps256_ps128(v0_3)));
        o3 += 2;
        _mm_storel_pd((double*) o2, _mm_castps_pd(_mm256_castps256_ps128(v0_2)));
        o2 += 2;
        _mm_storel_pd((double*) o1, _mm_castps_pd(_mm256_castps256_ps128(v0_1)));
        o1 += 2;
        _mm_storel_pd((double*) o0, _mm_castps_pd(_mm256_castps256_ps128(v0_0)));
        o0 += 2;
        v0_0 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_0), _mm256_castps_pd(v0_0)));
        v0_1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_1), _mm256_castps_pd(v0_1)));
        v0_2 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_2), _mm256_castps_pd(v0_2)));
        v0_3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_3), _mm256_castps_pd(v0_3)));
        v0_4 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_4), _mm256_castps_pd(v0_4)));
        v0_5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_5), _mm256_castps_pd(v0_5)));
        v0_6 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_6), _mm256_castps_pd(v0_6)));
        v0_7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(v0_7), _mm256_castps_pd(v0_7)));
      }
      if (bh & 1) {
        unaligned_store_u32(o7, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_7)));
        unaligned_store_u32(o6, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_6)));
        unaligned_store_u32(o5, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_5)));
        unaligned_store_u32(o4, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_4)));
        unaligned_store_u32(o3, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_3)));
        unaligned_store_u32(o2, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_2)));
        unaligned_store_u32(o1, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_1)));
        unaligned_store_u32(o0, _mm256_cvtsi256_si32(_mm256_castps_si256(v0_0)));
      }
    }

    i0 = (const float*) ((uintptr_t) i0 + input_reset);
    o0 = (float*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
