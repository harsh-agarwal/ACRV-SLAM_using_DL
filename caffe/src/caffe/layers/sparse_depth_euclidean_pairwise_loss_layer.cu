#include <vector>
#include <sstream>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_depth_euclidean_pairwise_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SetDiffZeroForMissingGTDepth(const int nthreads,
    const Dtype* const mask, const int num, 
    const int height, const int width, Dtype* const diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const mask_off = mask + offset;
    Dtype* const diff_off = diff + offset;
    
    // set diff = 0 based on mask
    Dtype mask_ = mask_off[0];

    if (mask_ == Dtype(0))
        diff_off[0] = Dtype(0);
  }
}

template <typename Dtype>
__global__ void SetBottomDiffZeroForMissingGTDepth(const int nthreads,
    const Dtype* const mask, const int num, 
    const int height, const int width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const mask_off = mask + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    
    // set bottom_diff = 0 based on mask
    Dtype mask_ = mask_off[0];

    if (mask_ == Dtype(0))
        bottom_diff_off[0] = Dtype(0);
  }
}

template <typename Dtype>
__global__ void ComputeDDiff(const int nthreads,
    const Dtype* const preds, const Dtype* const labels, const Dtype* const mask, const Dtype* const pairwise_weight, 
    const int num, const int height, const int width, Dtype* const ddiff_x, Dtype* const ddiff_y,
    Dtype* const ddiff_x2, Dtype* const ddiff_y2, Dtype* const mask_pairwise_x, Dtype* const mask_pairwise_y) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const preds_off = preds + offset;
    const Dtype* const labels_off = labels + offset;
    const Dtype* const mask_off = mask + offset;
    const Dtype* const pairwise_weight_off = pairwise_weight + offset;
    Dtype* const ddiff_x_off = ddiff_x + offset;
    Dtype* const ddiff_y_off = ddiff_y + offset;
    Dtype* const ddiff_x2_off = ddiff_x2 + offset;
    Dtype* const ddiff_y2_off = ddiff_y2 + offset;
    Dtype* const mask_pairwise_x_off = mask_pairwise_x + offset;
    Dtype* const mask_pairwise_y_off = mask_pairwise_y + offset;
    Dtype ddiff_x_val (0);
    Dtype ddiff_y_val (0);
    Dtype mask_val = *(mask_off);
    Dtype pairwise_weight_val = *(pairwise_weight_off);

    if (w < (width - 1)) 
    {
       if (*(mask_off + 1) != Dtype(0) && mask_val != Dtype(0)) 
       {    
          ddiff_x_val = *(preds_off + 1) - *(preds_off) - *(labels_off + 1) + *(labels_off);
          *(mask_pairwise_x_off) = Dtype(1);
       }
    }

    if (h < (height - 1)) 
    {
       if (*(mask_off + width) != Dtype(0) && mask_val != Dtype(0)) 
       {
          ddiff_y_val = *(preds_off + width) - *(preds_off) - *(labels_off + width) + *(labels_off);
          *(mask_pairwise_y_off) = Dtype(1);
       }
    }

    *(ddiff_x_off) = pairwise_weight_val * ddiff_x_val;
    *(ddiff_y_off) = pairwise_weight_val * ddiff_y_val;
    *(ddiff_x2_off) = pairwise_weight_val * ddiff_x_val * ddiff_x_val;
    *(ddiff_y2_off) = pairwise_weight_val * ddiff_y_val * ddiff_y_val;
  }
}

template <typename Dtype>
__global__ void ComputeDiv(const int nthreads,
    const Dtype N_pairwise_x, const Dtype N_pairwise_y, const int height, const int width, const Dtype* const ddiff_x, 
    const Dtype* const ddiff_y, Dtype* const bottom_diff, Dtype top_diff_val_1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const ddiff_x_off = ddiff_x + offset;
    const Dtype* const ddiff_y_off = ddiff_y + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    
    if (w > 0) *(bottom_diff_off) += top_diff_val_1*Dtype(2)/N_pairwise_x*(*(ddiff_x_off-1) - *(ddiff_x_off));
    else *(bottom_diff_off) += top_diff_val_1*Dtype(-2)/N_pairwise_x*(*ddiff_x_off);
    if (h > 0) *(bottom_diff_off) += top_diff_val_1*Dtype(2)/N_pairwise_y*(*(ddiff_y_off - width) - *(ddiff_y_off));
    else *(bottom_diff_off) += top_diff_val_1*Dtype(-2)/N_pairwise_y*(*ddiff_y_off);
  }
}

template <typename Dtype>
void SparseDepthEuclideanPairwiseLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int n_threads = num * height * width;
   
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  SetDiffZeroForMissingGTDepth<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[2]->gpu_data(), num, height, width, diff_.mutable_gpu_data());

  ComputeDDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), bottom[3]->gpu_data(), 
      num, height, width, ddiff_x_.mutable_gpu_data(), ddiff_y_.mutable_gpu_data(),
      ddiff_x2_.mutable_gpu_data(), ddiff_y2_.mutable_gpu_data(), mask_pairwise_x_.mutable_gpu_data(), mask_pairwise_y_.mutable_gpu_data());

  caffe_gpu_dot(count, mask_pairwise_x_.gpu_data(), ones_.gpu_data(), &N_pairwise_x_);
  caffe_gpu_dot(count, mask_pairwise_y_.gpu_data(), ones_.gpu_data(), &N_pairwise_y_);
  
  Dtype dot_tmp, ddiff_x2_sum_tmp, ddiff_y2_sum_tmp, diff_sum_tmp, N_tmp;

  Dtype dot(0);
  Dtype diff_sum2 (0);
  Dtype ddiff_x2_sum (0);
  Dtype ddiff_y2_sum (0);
  Dtype N2 (0);
  int offset = 0;

  for (int n = 0; n < num; ++n)
  {
     caffe_gpu_dot(height*width, diff_.gpu_data() + offset, diff_.gpu_data() + offset, &dot_tmp);
     caffe_gpu_dot(height*width, ddiff_x2_.gpu_data() + offset, ones_.gpu_data() + offset, &ddiff_x2_sum_tmp);
     caffe_gpu_dot(height*width, ddiff_y2_.gpu_data() + offset, ones_.gpu_data() + offset, &ddiff_y2_sum_tmp);
     caffe_gpu_dot(height*width, diff_.gpu_data() + offset, ones_.gpu_data() + offset, &diff_sum_tmp);
     caffe_gpu_dot(height*width, bottom[2]->gpu_data() + offset, ones_.gpu_data() + offset, &N_tmp);

     dot += N_tmp*dot_tmp;
     ddiff_x2_sum += ddiff_x2_sum_tmp;
     ddiff_y2_sum += ddiff_y2_sum_tmp;
     diff_sum2 += diff_sum_tmp*diff_sum_tmp;
     diff_sum_.mutable_cpu_data()[n] = diff_sum_tmp;
     N2 += N_tmp*N_tmp;
     N_.mutable_cpu_data()[n] = N_tmp;
     offset += height*width;
  }

  N2 = std::max(N2,Dtype(1));
  N_pairwise_x_ = std::max(N_pairwise_x_,Dtype(1));
  N_pairwise_y_ = std::max(N_pairwise_y_,Dtype(1));

  top[0]->mutable_cpu_data()[0] = dot / N2;
  top[1]->mutable_cpu_data()[0] = ddiff_x2_sum / N_pairwise_x_ + ddiff_y2_sum / N_pairwise_y_;
  top[2]->mutable_cpu_data()[0] = -diff_sum2 / N2;
}

template <typename Dtype>
void SparseDepthEuclideanPairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   
 if (propagate_down[0]) {

     int count = bottom[0]->count();
     int num = bottom[0]->num();
     int height = bottom[0]->height();
     int width = bottom[0]->width();
     int n_threads = num * height * width;

     Dtype N2 = Dtype(0);
     for (int n = 0; n < num; ++n) N2 += N_.cpu_data()[n]*N_.cpu_data()[n];
     N2 = std::max(N2,Dtype(1));

     Dtype top_diff_val_0 = top[0]->cpu_diff()[0];
     Dtype top_diff_val_1 = top[1]->cpu_diff()[0];
     Dtype top_diff_val_2 = top[2]->cpu_diff()[0];

     int offset = 0;

     for (int n = 0; n < num; ++n) 
     {
        caffe_gpu_axpby(
            height*width,              
            top_diff_val_0*Dtype(2)*N_.cpu_data()[n]/N2,      // a
            diff_.gpu_data() + offset,                        // x
            Dtype(0),                                         // b
            bottom[0]->mutable_gpu_diff() + offset);          // y

        Dtype diff_sum_tmp = diff_sum_.cpu_data()[n];

     	caffe_gpu_axpby(
            height*width,              
            top_diff_val_2*Dtype(-2)/N2*diff_sum_tmp,      // a
            ones_.gpu_data() + offset,                     // x
            Dtype(1),                                      // b
            bottom[0]->mutable_gpu_diff() + offset);       // y
        
        offset += height*width;
     }

     SetBottomDiffZeroForMissingGTDepth<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
         n_threads, bottom[2]->gpu_data(), num, height, width, bottom[0]->mutable_gpu_diff());
     
     ComputeDiv<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
         n_threads, N_pairwise_x_, N_pairwise_y_, height, width, ddiff_x_.gpu_data(), ddiff_y_.gpu_data(), bottom[0]->mutable_gpu_diff(), top_diff_val_1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseDepthEuclideanPairwiseLossLayer);

}  // namespace caffe
