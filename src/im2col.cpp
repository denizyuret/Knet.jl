#include <algorithm>

template <typename T>
void im2col(const T *img, T *col, int width, int height, int channels,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / (kernel_h * kernel_w);

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
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  std::fill(img, img + width*height*channels, 0);
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / (kernel_h * kernel_w);

    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h*stride_h - pad_h + h_offset;
        int w_pad = w*stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
          img[(c_im * height + h_pad) * width + w_pad] +=
            col[(c * height_col + h) * width_col + w];
        }
      }
    }
  }
}

extern "C" {
  void im2col_float(const float *img, float *col, int width, int height, int channels,
      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
    im2col(img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
  }
  void im2col_double(const double *img, double *col, int width, int height, int channels,
      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
    im2col(img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
  }

  void col2im_float(const float *col, float *img, int width, int height, int channels,
      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
    col2im(col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
  }
  void col2im_double(const double *col, double *img, int width, int height, int channels,
      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
    col2im(col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
  }
}
