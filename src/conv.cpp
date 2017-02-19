// Based on https://github.com/pluskid/Mocha.jl/tree/master/deps/pooling.cpp and im2col.cpp
// Modified by Deniz Yuret, 2017-02-18.
// Converted pooling backward pass to a maskless implementation.
// Added im2col mode argument to support conv (mode=0) and xcorr (mode=1).

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
		#pragma omp atomic
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
	      T ival = input[h * width + w];
	      #pragma omp atomic
              meanval += ival;
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
	      T oval = output[pool_index] / kernel_size;
	      int iidx = h * width + w;
	      #pragma omp atomic
              input[iidx] += oval;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void im2col(const T *img, T *col, int width, int height, int channels,
	    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  #pragma omp parallel for
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / (kernel_h * kernel_w);
    if (mode == 0) {
      w_offset = kernel_w - 1 - w_offset;
      h_offset = kernel_h - 1 - h_offset;
    }
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
	int h_pad = h*stride_h - pad_h + h_offset;
	int w_pad = w*stride_w - pad_w + w_offset;
	if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
	  col[(c*height_col+h) * width_col + w] =
	    img[(c_im * height + h_pad) * width + w_pad];
	} else {
	  col[(c*height_col+h) * width_col + w] = 0;
	}
      }
    }
  }
}


template <typename T>
void col2im(const T *col, T *img, int width, int height, int channels,
	    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  memset(img, 0, width*height*channels*sizeof(T));
  #pragma omp parallel for
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / (kernel_h * kernel_w);
    if (mode == 0) {
      w_offset = kernel_w - 1 - w_offset;
      h_offset = kernel_h - 1 - h_offset;
    }
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
	int h_pad = h*stride_h - pad_h + h_offset;
	int w_pad = w*stride_w - pad_w + w_offset;
	if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
	  T cval = col[(c * height_col + h) * width_col + w];
	  int iidx = (c_im * height + h_pad) * width + w_pad;
          #pragma omp atomic
	  img[iidx] += cval;
	}
      }
    }
  }
}


extern "C" {

void max_pooling_fwd32(const float* global_input, float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_fwd(global_input, global_output,
      width, height, channels, num,
      pooled_width, pooled_height,
      kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void max_pooling_fwd64(const double* global_input, double *global_output, 
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_fwd(global_input, global_output, 
      width, height, channels, num,
      pooled_width, pooled_height,
      kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void max_pooling_bwd32(float* global_input, const float *global_output, const float *grad_output, float *grad_input,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_bwd(global_input, global_output, grad_output, grad_input,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void max_pooling_bwd64(double* global_input, const double *global_output, const double *grad_output, double *grad_input,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  max_pooling_bwd(global_input, global_output, grad_output, grad_input,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void mean_pooling_fwd32(const float* global_input, float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_fwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void mean_pooling_fwd64(const double* global_input, double *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_fwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void mean_pooling_bwd32(float* global_input, const float *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_bwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}
void mean_pooling_bwd64(double* global_input, const double *global_output,
    int width, int height, int channels, int num,
    int pooled_width, int pooled_height,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
  mean_pooling_bwd(global_input, global_output,
    width, height, channels, num, pooled_width, pooled_height,
    kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

void im2col32(const float *img, float *col, int width, int height, int channels,
		    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {
  im2col(img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, mode);
}
void im2col64(const double *img, double *col, int width, int height, int channels,
		     int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {
  im2col(img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, mode);
}

void col2im32(const float *col, float *img, int width, int height, int channels,
	      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {
  col2im(col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, mode);
}
void col2im64(const double *col, double *img, int width, int height, int channels,
	      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int mode) {
  col2im(col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, mode);
}

} // extern "C"
