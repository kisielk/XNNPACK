// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/indirection.h>
#include <xnnpack/operator.h>

#include <gtest/gtest.h>

class IndirectionTester {
 public:
  IndirectionTester& input_height(size_t input_height) {
    input_height_ = input_height;
    return *this;
  }

  IndirectionTester& input_width(size_t input_width) {
    input_width_ = input_width;
    return *this;
  }

  IndirectionTester& kernel_height(size_t kernel_height) {
    kernel_height_ = kernel_height;
    return *this;
  }

  IndirectionTester& kernel_width(size_t kernel_width) {
    kernel_width_ = kernel_width;
    return *this;
  }

  IndirectionTester& padding_height(size_t padding_height) {
    padding_height_ = padding_height;
    return *this;
  }

  IndirectionTester& padding_width(size_t padding_width) {
    padding_width_ = padding_width;
    return *this;
  }

  IndirectionTester& subsampling(size_t subsampling) {
    subsampling_ = subsampling;
    return *this;
  }

  IndirectionTester& dilation(size_t dilation) {
    dilation_ = dilation;
    return *this;
  }

  IndirectionTester& channels(size_t channels) {
    channels_ = channels;
    return *this;
  }

  IndirectionTester& primary_tile(size_t primary_tile) {
    primary_tile_ = primary_tile;
    return *this;
  }

  IndirectionTester& channel_tile(size_t channel_tile) {
    channel_tile_ = channel_tile;
    return *this;
  }

  IndirectionTester& expected_indices(std::vector<size_t> expected_indices) {
    expected_indices_ = expected_indices;
    return *this;
  }

  void Test() {
    IndirectionInit();
    for (size_t i = 0; i < expected_indices_.size(); i++) {
      EXPECT_EQ(indirection_buffer_[i], &input_[expected_indices_[i]]) << i << expected_indices_[i];
    }
  }

 private:
  void IndirectionInit() {
    const size_t kernel_size = kernel_height_ * kernel_width_;
    const size_t output_height = xnn_compute_convolution_output_dimension(
        input_height_ + padding_height_, kernel_height_, dilation_, subsampling_);
    const size_t output_width = xnn_compute_convolution_output_dimension(
        input_width_ + padding_width_, kernel_width_, dilation_, subsampling_);
    const size_t step_width = dilation_ == 1 ? subsampling_ : kernel_width_;
    const size_t step_height =
        kernel_size + (output_width - 1) * step_width * kernel_height_;

    input_ = std::vector<float>(channels_ * input_height_ * input_width_ +
                         XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero_buffer(channels_ + XNN_EXTRA_BYTES / sizeof(float));

    const size_t num_indirection_elements = output_height * step_height;
    indirection_buffer_ = std::vector<const float*>(num_indirection_elements);
    xnn_operator op = {};
    op.indirection_buffer = reinterpret_cast<const void**>(indirection_buffer_.data());
    op.input = input_.data();
    op.input_pixel_stride = channels_;
    op.zero_buffer = zero_buffer.data();
    op.input_height = input_height_;
    op.input_width = input_width_;
    op.output_height = output_height;
    op.output_width = output_width;
    op.kernel_height = kernel_height_;
    op.kernel_width = kernel_width_;
    op.stride_height = subsampling_;
    op.stride_width = subsampling_;
    op.dilation_height = dilation_;
    op.dilation_width = dilation_;
    op.padding_top = padding_height_ / 2;
    op.padding_left = padding_width_ / 2;
    xnn_indirection_init_dwconv2d(&op, step_height, step_width, primary_tile_, /*log2_element_size*/2);
  }

  // Set by tests using setter functions.
  size_t input_height_;
  size_t input_width_;
  size_t kernel_height_;
  size_t kernel_width_;
  size_t padding_height_;
  size_t padding_width_;
  size_t subsampling_;
  size_t dilation_;
  size_t channels_;
  size_t primary_tile_;
  size_t channel_tile_;
  std::vector<size_t> expected_indices_;

  // Initialized by IndirectionInit.
  std::vector<const float*> indirection_buffer_;
  std::vector<float> input_;
};

TEST(INDIRECTION, input3x3_kernel1x1) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(1)
      .kernel_width(1)
      .padding_height(0)
      .padding_width(0)
      .subsampling(1)
      .dilation(1)
      .channels(1)
      .primary_tile(1)
      .channel_tile(8)
      .expected_indices({0, 1, 2, 3, 4, 5, 6, 7, 8})
      .Test();
}

TEST(INDIRECTION, input3x3_kernel2x2) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(2)
      .kernel_width(2)
      .padding_height(0)
      .padding_width(0)
      .subsampling(1)
      .dilation(1)
      .channels(1)
      .primary_tile(4)
      .channel_tile(8)
      // input:  kernel:
      // 0 1 2   a b
      // 3 4 5   c d
      // 6 7 8
      .expected_indices({
        // For each output row, column major, and compress pointers within a single output row.
        0, 3, 1, 4, 2, 5,
        3, 6, 4, 7, 5, 8,
      })
      .Test();
}

// TODO(zhin): add tests for primary_tile == kernel_size with subsampling.
// TODO(zhin): add tests for primary_tile < kernel_size.
// TODO(zhin): add tests for primary_tile < kernel_size with subsampling.
