#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_depth_euclidean_pairwise_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SparseDepthEuclideanPairwiseLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  diff_.ReshapeLike(*bottom[0]);
  ddiff_x_.ReshapeLike(*bottom[0]);
  ddiff_y_.ReshapeLike(*bottom[0]);
  ddiff_x2_.ReshapeLike(*bottom[0]);
  ddiff_y2_.ReshapeLike(*bottom[0]);
  mask_pairwise_x_.ReshapeLike(*bottom[0]);
  mask_pairwise_y_.ReshapeLike(*bottom[0]);
  caffe_set(mask_pairwise_x_.count(), Dtype(0), mask_pairwise_x_.mutable_cpu_data());
  caffe_set(mask_pairwise_y_.count(), Dtype(0), mask_pairwise_y_.mutable_cpu_data());
  diff_sum_.Reshape(vector<int>(1, bottom[0]->num()));
  N_.Reshape(vector<int>(1, bottom[0]->num()));
  ones_.ReshapeLike(*bottom[0]);
  caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
  top[1]->ReshapeLike(*top[0]);
  top[2]->ReshapeLike(*top[0]);
}

template <typename Dtype>
void SparseDepthEuclideanPairwiseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height * width;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* mask = bottom[2]->cpu_data();
  const Dtype* pairwise_weight = bottom[3]->cpu_data();
  Dtype* diff = diff_.mutable_cpu_data();  
  Dtype* ddiff_x = ddiff_x_.mutable_cpu_data();  
  Dtype* ddiff_y = ddiff_y_.mutable_cpu_data();

  caffe_sub(
      count,
      bottom_data,
      label,
      diff_.mutable_cpu_data());

  // set diff = 0 based on mask
  for (int n = 0; n < num; ++n)
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask_ = *(mask + bottom[2]->offset(n) + i);	
       if (mask_ == Dtype(0))
	  *(diff + bottom[1]->offset(n) + i) = Dtype(0);
    }
  }

  Dtype dot (0);
  Dtype diff_sum2 (0);
  Dtype ddiff_x2_sum (0);
  Dtype ddiff_y2_sum (0);
  Dtype N2 (0);
  N_pairwise_x_ = Dtype(0);
  N_pairwise_y_ = Dtype(0);

  int offset = 0;

  for (int n = 0; n < num; ++n)
  {
     Dtype dot_tmp = caffe_cpu_dot(height*width, diff_.cpu_data() + offset, diff_.cpu_data() + offset);
     Dtype diff_sum_tmp = caffe_cpu_dot(height*width, diff_.cpu_data() + offset, ones_.cpu_data() + offset);
     Dtype N_tmp = caffe_cpu_dot(height*width, bottom[2]->cpu_data() + offset, ones_.cpu_data() + offset);
     diff_sum_.mutable_cpu_data()[n] = diff_sum_tmp;
     diff_sum2 += diff_sum_tmp*diff_sum_tmp;
     dot += N_tmp*dot_tmp;
     N_.mutable_cpu_data()[n] = N_tmp;
     N2 += N_tmp*N_tmp;
     offset += height*width;
  }
    
  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
        if (i%width < (width - 1))
        {
           Dtype diff_pred = *(bottom_data + bottom[0]->offset(n) + i + 1) - *(bottom_data + bottom[0]->offset(n) + i);
           Dtype diff_gt (0);
           Dtype mask_val1 = *(mask + bottom[2]->offset(n) + i + 1);
           Dtype mask_val2 = *(mask + bottom[2]->offset(n) + i);
           Dtype pairwise_weight_val = *(pairwise_weight + bottom[3]->offset(n) + i);
           Dtype label_val1 = *(label + bottom[1]->offset(n) + i + 1);
           Dtype label_val2 = *(label + bottom[1]->offset(n) + i);
           Dtype diff_tmp (0);

           if (mask_val1 != Dtype(0) && mask_val2 != Dtype(0))
           {
              diff_gt = label_val1 - label_val2;
              diff_tmp = diff_pred - diff_gt;
              ++N_pairwise_x_;
           }
            
           *(ddiff_x + bottom[1]->offset(n) + i) = pairwise_weight_val*diff_tmp;
           ddiff_x2_sum += pairwise_weight_val*diff_tmp*diff_tmp;
        }
	else *(ddiff_x + bottom[1]->offset(n) + i) = Dtype(0);
        if (i < (height-1)*width) 
        {
           Dtype diff_pred = *(bottom_data + bottom[0]->offset(n) + i + width) - *(bottom_data + bottom[0]->offset(n) + i);                    
           Dtype diff_gt (0);
           Dtype mask_val1 = *(mask + bottom[2]->offset(n) + i + width);
           Dtype mask_val2 = *(mask + bottom[2]->offset(n) + i);
           Dtype pairwise_weight_val = *(pairwise_weight + bottom[3]->offset(n) + i);
           Dtype label_val1 = *(label + bottom[1]->offset(n) + i + width);
           Dtype label_val2 = *(label + bottom[1]->offset(n) + i);     
           Dtype diff_tmp (0);

           if (mask_val1 != Dtype(0) && mask_val2 != Dtype(0)) 
           {
              diff_gt = label_val1 - label_val2;
              diff_tmp = diff_pred - diff_gt;
              ++N_pairwise_y_;
           }

           *(ddiff_y + bottom[1]->offset(n) + i) = pairwise_weight_val*diff_tmp;
           ddiff_y2_sum += pairwise_weight_val*diff_tmp*diff_tmp;
        }
        else *(ddiff_y + bottom[1]->offset(n) + i) = Dtype(0);
    }
  }
  
  N2 = std::max(N2,Dtype(1));
  N_pairwise_x_ = std::max(N_pairwise_x_,Dtype(1));
  N_pairwise_y_ = std::max(N_pairwise_y_,Dtype(1));
  top[0]->mutable_cpu_data()[0] = dot / N2;
  top[1]->mutable_cpu_data()[0] = ddiff_x2_sum / N_pairwise_x_ + ddiff_y2_sum / N_pairwise_y_;
  top[2]->mutable_cpu_data()[0] = -diff_sum2 / N2;
}

template <typename Dtype>
void SparseDepthEuclideanPairwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

   if (propagate_down[0]) {
     int num = bottom[0]->num();
     int height = bottom[0]->height();
     int width = bottom[0]->width();
     int spatial_count = height * width;
     const Dtype* mask = bottom[2]->cpu_data();
     const Dtype* diff = diff_.cpu_data();
     const Dtype* ddiff_x = ddiff_x_.cpu_data();
     const Dtype* ddiff_y = ddiff_y_.cpu_data();
     Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); 
     Dtype top_diff_val_0 = top[0]->cpu_diff()[0];
     Dtype top_diff_val_1 = top[1]->cpu_diff()[0];
     Dtype top_diff_val_2 = top[2]->cpu_diff()[0];

     Dtype N2 (0);
     for (int n = 0; n < num; ++n) N2 += N_.cpu_data()[n]*N_.cpu_data()[n];
     N2 = std::max(N2,Dtype(1));

     for (int n = 0; n < num; ++n) 
     {
        for (int i = 0; i < spatial_count; ++i) 
        {
           Dtype mask_ = *(mask + bottom[2]->offset(n) + i);	
           if (mask_ == Dtype(0))
	      *(bottom_diff + bottom[0]->offset(n) + i) = Dtype(0);
           else 
           {
              Dtype diff_val = *(diff + bottom[0]->offset(n) + i);
	      *(bottom_diff + bottom[0]->offset(n) + i) = top_diff_val_0*Dtype(2)*N_.cpu_data()[n]/N2*diff_val;
              *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val_2*Dtype(-2)/N2*diff_sum_.cpu_data()[n];
           }
           if (i%width != 0)
              *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val_1*Dtype(2)/N_pairwise_x_*(*(ddiff_x + bottom[0]->offset(n) + i - 1) - *(ddiff_x + bottom[0]->offset(n) + i));
           else *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val_1*Dtype(-2)/N_pairwise_x_*(*(ddiff_x + bottom[0]->offset(n) + i));
           if (i >= width)
              *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val_1*Dtype(2)/N_pairwise_y_*(*(ddiff_y + bottom[0]->offset(n) + i - width) - *(ddiff_y + bottom[0]->offset(n) + i));
           else
              *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val_1*Dtype(-2)/N_pairwise_y_*(*(ddiff_y + bottom[0]->offset(n) + i));
       }
     }
   }
}

#ifdef CPU_ONLY
STUB_GPU(SparseDepthEuclideanPairwiseLossLayer);
#endif

INSTANTIATE_CLASS(SparseDepthEuclideanPairwiseLossLayer);
REGISTER_LAYER_CLASS(SparseDepthEuclideanPairwiseLoss);

}  // namespace caffe

