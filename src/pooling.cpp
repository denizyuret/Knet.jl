// Based on https://github.com/pluskid/Mocha.jl/tree/master/deps/pooling.cpp
// Modified by Deniz Yuret, 2017-02-18.
// Converted backward pass to a maskless implementation.

#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdio>

template <typename T>
void max_pooling_fwd(const T* global_input, T *global_output, 
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int input_offset = width*height;
  int output_offset = pooled_width*pooled_height;

  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int offset = (n * channels + c);
      const T *input = global_input + input_offset * offset;
      T *output = global_output + output_offset * offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph*stride_h - pad_h;
          int wstart = pw*stride_w - pad_w;
          int hend   = std::min(hstart + kernel_h, height);
          int wend   = std::min(wstart + kernel_w, width);
	  hstart = std::max(hstart, 0);
	  wstart = std::max(wstart, 0);

          int pool_index = ph * pooled_width + pw;
          T maxval = -std::numeric_limits<T>::max();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int index = h * width + w;
              if (input[index] > maxval) {
                maxval = input[index];
              }
            }
          }
          output[pool_index] = maxval;
        }
      }
    }
  }
}

template <typename T>
void max_pooling_bwd(const T* global_input, const T *global_output, const T* grad_output, T* grad_input,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int input_offset = width*height;
  int output_offset = pooled_width*pooled_height;
  memset(grad_input, 0, input_offset*channels*num*sizeof(T));

  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int offset = (n * channels + c);
      int offset_i = input_offset * offset;
      int offset_o = output_offset * offset;
      const T *input = global_input + offset_i;
      const T *output = global_output + offset_o;
      const T *d_output = grad_output + offset_o;
      T *d_input = grad_input + offset_i;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph*stride_h - pad_h;
          int wstart = pw*stride_w - pad_w;
          int hend   = std::min(hstart + kernel_h, height);
          int wend   = std::min(wstart + kernel_w, width);
	  hstart = std::max(hstart, 0);
	  wstart = std::max(wstart, 0);

          int pool_index = ph * pooled_width + pw;
	  T maxval = output[pool_index];
	  T d_maxval = d_output[pool_index];
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int index = h * width + w;
              if (input[index] == maxval) {
		d_input[index] += d_maxval;
              }
            }
          }
        }
      }
    }
  }
}


template <typename T>
void mean_pooling_fwd(const T* global_input, T *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int input_offset = width*height;
  int output_offset = pooled_width*pooled_height;
  int kernel_size = kernel_w * kernel_h;

  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int offset = (n * channels + c);
      const T *input = global_input + input_offset * offset;
      T *output = global_output + output_offset * offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph*stride_h - pad_h;
          int wstart = pw*stride_w - pad_w;
          int hend   = std::min(hstart + kernel_h, height);
          int wend   = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);

          int pool_index = ph * pooled_width + pw;
          T meanval = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              meanval += input[h * width + w];
            }
          }
          output[pool_index] = meanval / kernel_size;
        }
      }
    }
  }
}

template <typename T>
void mean_pooling_bwd(T* global_input, const T *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int input_offset = width*height;
  int output_offset = pooled_width*pooled_height;
  int kernel_size = kernel_w * kernel_h;
  memset(global_input, 0, input_offset*channels*num*sizeof(T));

  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int offset = (n * channels + c);
      T *input = global_input + input_offset * offset;
      const T *output = global_output + output_offset * offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph*stride_h - pad_h;
          int wstart = pw*stride_w - pad_w;
          int hend   = std::min(hstart + kernel_h, height);
          int wend   = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);

          int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              input[h * width + w] += output[pool_index] / kernel_size;
            }
          }
        }
      }
    }
  }
}

extern "C" {

void max_pooling_fwd_float(const float* global_input, float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_fwd(global_input, global_output,
      width, height, channels, num,
      pooled_width, pooled_height,
      kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void max_pooling_fwd_double(const double* global_input, double *global_output, 
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_fwd(global_input, global_output, 
      width, height, channels, num,
      pooled_width, pooled_height,
      kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void max_pooling_bwd_float(float* global_input, const float *global_output, const float *grad_output, float *grad_input,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_bwd(global_input, global_output, grad_output, grad_input,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void max_pooling_bwd_double(double* global_input, const double *global_output, const double *grad_output, double *grad_input,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_bwd(global_input, global_output, grad_output, grad_input,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void mean_pooling_fwd_float(const float* global_input, float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_fwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void mean_pooling_fwd_double(const double* global_input, double *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_fwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void mean_pooling_bwd_float(float* global_input, const float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_bwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void mean_pooling_bwd_double(double* global_input, const double *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_bwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

} // extern "C"
