#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_log_layer.hpp"
#define OUT_VAL Dtype(-9.210340372)

namespace caffe {

template <typename Dtype>
__global__ void ComputeLogDepths(const int nthreads,
    const Dtype* const bottom_data, const int num, 
    const int height, const int width, Dtype* const top, 
    Dtype scale, Dtype shift, Dtype out_val) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const bottom_data_off = bottom_data + offset;
    Dtype* const top_off = top + offset;
    
    Dtype mask = bottom_data_off[0];

    if (mask > Dtype(0))
        top_off[0] = scale*log(mask) + shift;
    else
        top_off[0] = out_val;
  }
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int n_threads = num * height * width;
  ComputeLogDepths<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), num, height, width, top[0]->mutable_gpu_data(), 
      scale_, shift_, OUT_VAL);
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseLogLayer);

}  // namespace caffe
