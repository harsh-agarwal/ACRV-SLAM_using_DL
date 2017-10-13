#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_log_layer.hpp"
#define OUT_VAL Dtype(-9.210340372)

namespace caffe {

template <typename Dtype>
void SparseLogLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  scale_ = this->layer_param_.sparse_log_param().scale();
  shift_ = this->layer_param_.sparse_log_param().shift();
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height * width;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_ = top[0]->mutable_cpu_data();

  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(bottom_data + bottom[0]->offset(n) + i);	
       if (mask > Dtype(0))
	  *(top_ + bottom[0]->offset(n) + i) = scale_*log(mask) + shift_;
       else
          *(top_ + bottom[0]->offset(n) + i) = OUT_VAL;
    }
  }
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(SparseLogLayer);
#endif

INSTANTIATE_CLASS(SparseLogLayer);
REGISTER_LAYER_CLASS(SparseLog);

}  // namespace caffe
