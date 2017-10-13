#include <vector>

#include "caffe/layers/data_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#define PI 3.14159265
#define IM_LABEL_RATIO 2.0

namespace caffe {

std::vector<int> GetCropDims(int crop_height, int crop_width, int height, int width, int h_off, int w_off) 
{
   std::vector<int> crop_dims; 
   int centre_x = width/2.;
   int centre_y = height/2.;
   int centre_crop_x = crop_width/2.;
   int centre_crop_y = crop_height/2.;
   int crop_w_start = centre_x - centre_crop_x - w_off;
   int crop_w_end = centre_x + centre_crop_x - w_off + crop_width%2;
   int crop_h_start = centre_y - centre_crop_y - h_off;
   int crop_h_end = centre_y + centre_crop_y - h_off + crop_width%2;
   
   // size checks
   CHECK_GE(crop_w_start, 0);
   CHECK_LE(crop_w_end, width);
   CHECK_GE(crop_h_start, 0);
   CHECK_LE(crop_h_end, height); 
   CHECK_EQ(crop_w_end - crop_w_start, crop_width);
   CHECK_EQ(crop_h_end - crop_h_start, crop_height);
   // end of size checks

   crop_dims.push_back(crop_w_start);
   crop_dims.push_back(crop_w_end);
   crop_dims.push_back(crop_h_start);
   crop_dims.push_back(crop_h_end);

   return crop_dims;
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::InitRand() 
{
   const unsigned int rng_seed = caffe_rng_rand();
   rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int DataAugmentationLayer<Dtype>::Rand(int n) 
{
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    InitRand();
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{    
    const DataAugmentationParameter& param_ = this->layer_param_.data_augmentation_param();
    CHECK_LE (static_cast<int>(param_.gen_depths_scale()), static_cast<int>(param_.gen_depths_sparse_labels()));
   
    int num_ = static_cast<int>(bottom[0]->num());
    sparse_count = 0; 
    for (int i=0; i < bottom.size(); ++i) 
    {
       int channels_ = static_cast<int>(bottom[i]->channels());
       int crop_height_ = param_.crop_height(i);
       int crop_width_ = param_.crop_width(i);
       top[i]->Reshape(num_, channels_, crop_height_, crop_width_);
       if (param_.data_type(i) == DataAugmentationParameter_DataType_DEPTHS_SPARSE)
	  sparse_count++;
       if (param_.gen_depths_sparse_labels() && param_.data_type(i) == DataAugmentationParameter_DataType_DEPTHS_LABELS)
       {
	  for (int ii=0; ii < sparse_count; ii++)
	  {
	      top[bottom.size()+ii]->Reshape(num_, channels_, crop_height_, crop_width_);
	      caffe_set(top[bottom.size()+ii]->count(), Dtype(0), top[bottom.size()+ii]->mutable_cpu_data());
	  }
       }
    }       
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{    
    /* 
    Transform types 
    all_ == 1        : Apply all random transformations below
    scale_ == 1      : Scale: Input and target images are scaled by s ∈ [1, 1.5], and the depths are divided by s. (default: s = 1)
    rotate_ == 1     : Rotation: Input and target are rotated by r ∈ [−5, 5] degrees. (default: r = 0 degrees)
    translate_ == 1  : Translation: Input and target are randomly cropped to the sizes specified by crop_height and crop_width (default: centre crop)
    color_ == 1      : Color: Input values are multiplied globally by a random RGB value c ∈ [0.8, 1.2]^3 (default: c = [1]^3) 
    flip_ == 1       : Flips: Input and target are horizontally flipped with 0.5 probability. (default: false)
    */

    const DataAugmentationParameter& param_ = this->layer_param_.data_augmentation_param();
    int num_ = static_cast<int>(bottom[0]->num());
    int height_[bottom.size()];
    int width_[bottom.size()];
    int crop_height_[bottom.size()];
    int crop_width_[bottom.size()];
    if (param_.gen_depths_sparse_labels()) 
       caffe_set(top[bottom.size()]->count(), Dtype(0), top[bottom.size()]->mutable_cpu_data()); // assumes top[bottom.size()] store depths_sparse_labels

    for (int i = 0; i < bottom.size(); ++i) 
    {
        crop_height_[i] = param_.crop_height(i);
        crop_width_[i] = param_.crop_width(i);
        height_[i] = static_cast<int>(bottom[i]->height());
        width_[i] = static_cast<int>(bottom[i]->width());
    }
    
    Dtype s_ = Dtype(1), r_ = Dtype(0), h_off_ = Dtype(0), w_off_ = Dtype(0), c_R_ = Dtype(1), c_G_ = Dtype(1), c_B_ = Dtype(1);
    bool f_ = false;
    Dtype rot_coeff_[6] = {0};
    int N_p = 100/(param_.p()*100);
    bool phase_train_ = this->phase_ == TRAIN;

    for (int n = 0; n < num_; ++n) 
    {
      if (phase_train_ && Rand(N_p) == 0) 
      {	      
	 if (param_.all() || param_.scale() ) 
         {
	     s_ = Dtype(1) + static_cast<Dtype>(Rand(51))/Dtype(100);
	     //std::cout << "s_: " << s_ << std::endl;
	 } 

	 if (param_.all() || param_.rotate() )  
         {
	    r_ = static_cast<Dtype>(Rand(101))/Dtype(10) - Dtype(5);
	    //std::cout << "r_: " << r_ << std::endl;
	 } 

	 if (param_.all() || param_.translate() ) 
         {
	    Dtype r =  Rand(11)+1;
	    Dtype r2 =  Rand(14)+1;
	    h_off_ = r - 6;
	    w_off_ = r2 -8;
	    //std::cout << "h_off_: " << h_off_ << " w_off_: " << w_off_ << std::endl;
	 }  

	 if (param_.all() || param_.color() ) 
         {
	    c_R_ = Dtype(0.8) + static_cast<Dtype>(Rand(41))/Dtype(100);
	    c_G_ = Dtype(0.8) + static_cast<Dtype>(Rand(41))/Dtype(100);
	    c_B_ = Dtype(0.8) + static_cast<Dtype>(Rand(41))/Dtype(100); 
            //std::cout << "c_R_: " << c_R_ << " c_G_: " << c_G_ << " c_B_: " << c_B_ << std::endl;
	 }  
	      
	 if (param_.all() || param_.flip() ) 
         {
	    f_ = static_cast<bool>(Rand(2)); 
	    //std::cout << "f_: " << f_ << std::endl;
	 }
      } 
  
      int sparse_processed= 0;

      for (int index = 0; index < bottom.size(); ++index) 
      {  
          const Dtype *bottom_pointer = bottom[index]->cpu_data();
          Dtype *top_pointer = top[index]->mutable_cpu_data();
          Dtype s__;
          if (param_.data_type(index) == DataAugmentationParameter_DataType_IMAGE_OUT) 
             s__ = s_/Dtype(IM_LABEL_RATIO);
          else
             s__ = s_;

          int resized_height_ = Dtype(height_[index])*s__;
          int resized_width_ =  Dtype(width_[index])*s__;

          vector <int> crop_dims;
          if (param_.data_type(index) == DataAugmentationParameter_DataType_DEPTHS_LABELS) 
               crop_dims = GetCropDims(crop_height_[index], crop_width_[index], resized_height_, resized_width_, h_off_/Dtype(IM_LABEL_RATIO), w_off_/Dtype(IM_LABEL_RATIO));
          else if (param_.data_type(index) == DataAugmentationParameter_DataType_IMAGE_OUT)
               crop_dims = GetCropDims(crop_height_[index], crop_width_[index], resized_height_, resized_width_, h_off_/Dtype(IM_LABEL_RATIO), w_off_/Dtype(IM_LABEL_RATIO));
          else
               crop_dims = GetCropDims(crop_height_[index], crop_width_[index], resized_height_, resized_width_, h_off_, w_off_);


          /// start of im processing
          if (param_.data_type(index) == DataAugmentationParameter_DataType_IMAGE || param_.data_type(index) == DataAugmentationParameter_DataType_IMAGE_OUT) 
          {  
             cv::Mat bottom_mat, bottom_resized_mat, top_mat;
             bottom_mat = cv::Mat(height_[index], width_[index], CV_8UC3);
             uchar *bottom_mat_pointer = (uchar*) bottom_mat.data;
             bottom_resized_mat = cv::Mat(resized_height_, resized_width_, CV_8UC3);
             top_mat = bottom_resized_mat.colRange(crop_dims[0], crop_dims[1]).rowRange(crop_dims[2], crop_dims[3]);
            
             int ii = 0;           
             for (int i = 0; i < height_[index]; ++i) 
             {
                 for (int j = 0; j < width_[index]; ++j) 
                 {
                     int j_;
                     if (f_) j_ = width_[index] - j - 1;
                     else    j_ = j;  	

                     bottom_mat_pointer[ii    ] = static_cast<uchar>(std::min(Dtype(255),c_R_* *(bottom_pointer + bottom[index]->offset(n,0,i) + j_)));
		     bottom_mat_pointer[ii + 1] = static_cast<uchar>(std::min(Dtype(255),c_G_* *(bottom_pointer + bottom[index]->offset(n,1,i) + j_)));
		     bottom_mat_pointer[ii + 2] = static_cast<uchar>(std::min(Dtype(255),c_B_* *(bottom_pointer + bottom[index]->offset(n,2,i) + j_)));
                     ii+=3;
                 }
             } 
             
             if (r_ != Dtype(0)) 
             {  
                cv::Point2f center((Dtype)height_[index]/2., (Dtype)width_[index]/2.);
                cv::Mat rot_mat = cv::getRotationMatrix2D(center, r_, 1.0);
                cv::warpAffine(bottom_mat, bottom_mat, rot_mat, bottom_mat.size());
             }

             cv::resize(bottom_mat, bottom_resized_mat, bottom_resized_mat.size(), s__, cv::INTER_LINEAR);
             //cv::imshow("im", bottom_resized_mat);
             //cv::waitKey(0);
            
             for (int i = 0; i < crop_height_[index]; ++i) 
             {
                 for (int j = 0; j < crop_width_[index]; ++j) 
                 {

                     *(top_pointer + top[index]->offset(n,0,i) + j) = static_cast<Dtype>(top_mat.at<cv::Vec3b>(i,j)[0]);
                     *(top_pointer + top[index]->offset(n,1,i) + j) = static_cast<Dtype>(top_mat.at<cv::Vec3b>(i,j)[1]);
                     *(top_pointer + top[index]->offset(n,2,i) + j) = static_cast<Dtype>(top_mat.at<cv::Vec3b>(i,j)[2]);
                 }
             }                                   
          } // end of im processing

          
          /// start of depths processing
          if ((param_.data_type(index) == DataAugmentationParameter_DataType_DEPTHS_SPARSE) || 
             (param_.data_type(index) == DataAugmentationParameter_DataType_DEPTHS_LABELS) ) 
          {
             cv::Mat bottom_mat, bottom_resized_mat, top_mat;
             bottom_mat = cv::Mat(height_[index], width_[index], CV_32F);
             float *bottom_mat_pointer = (float*) bottom_mat.data;
             bottom_resized_mat = cv::Mat(resized_height_, resized_width_, CV_32F);
             top_mat = bottom_resized_mat.colRange(crop_dims[0], crop_dims[1]).rowRange(crop_dims[2], crop_dims[3]);

             int ii = 0;   
    
             for (int i=0; i < height_[index]; ++i) 
             {
                 for (int j = 0; j < width_[index]; ++j) 
                 {
                     int j_;
                     if (f_) 
                        j_ = width_[index] - j - 1;
                     else    
                        j_ = j;  	

		     Dtype depth_val = static_cast<float>(*(bottom_pointer + bottom[index]->offset(n,0,i) + j_));
                     bottom_mat_pointer[ii] = depth_val;
                     ++ii;
                 }
             } 
                        
             if (r_ != Dtype(0)) 
             {
                cv::Point2f center((Dtype)resized_height_/2., (Dtype)resized_width_/2.);
                Dtype alpha_ = cos(Dtype(-1)*r_* PI / 180.0), beta_ = sin(Dtype(-1)*r_* PI / 180.0);
                rot_coeff_[0] = alpha_;
                rot_coeff_[1] = beta_;
                rot_coeff_[2] = (1-alpha_)*center.x - beta_*center.y;
                rot_coeff_[3] = -beta_;
                rot_coeff_[4] = alpha_;
                rot_coeff_[5] = beta_*center.x - (1-alpha_)*center.y;
             }
               
             float *bottom_resized_mat_pointer = (float*) bottom_resized_mat.data;
             ii = 0;
             for (int i = 0; i < resized_height_; ++i) 
             {
                 for (int j = 0; j < resized_width_; ++j) 
                 { 
                     Dtype x_src, y_src;
                     if (r_ != Dtype(0)) 
                     {
		        x_src = (rot_coeff_[0]*Dtype(j) + rot_coeff_[1]*Dtype(i) + rot_coeff_[2])/s_;
                        y_src = (rot_coeff_[3]*Dtype(j) + rot_coeff_[4]*Dtype(i) + rot_coeff_[5])/s_;
                     }
                     else 
                     {
                        x_src = (Dtype) j/s_;
                        y_src = (Dtype) i/s_;
                     }

                     if (x_src < 0 || x_src >= width_[index] || y_src < 0 || y_src >= height_[index]) 
                        bottom_resized_mat_pointer[ii] = 0;
                     else 
                        bottom_resized_mat_pointer[ii] = bottom_mat.at<float>((int) y_src, (int) x_src);                  
                     ++ii;    
                 }
             }
            
             bool flag1_ = false;
             if (param_.data_type(index) == DataAugmentationParameter_DataType_DEPTHS_SPARSE && param_.gen_depths_sparse_labels()) 
             { 
                sparse_processed++;
            	flag1_ = true;    
             }
                  
             for (int i = 0; i < crop_height_[index]; ++i) 
             {
                 for (int j = 0; j < crop_width_[index]; ++j) 
                 {
		     Dtype depth_val = static_cast<Dtype>(top_mat.at<float>(i,j));
		     if (sparse_processed != sparse_count)
                        depth_val = depth_val/s_;

                     if (flag1_ && depth_val > Dtype(0) && j>=4 && j<crop_width_[index]-6 && i>=4 && i<crop_height_[index]-6) 
                     {
                        Dtype x_dst = Dtype(j-4)/Dtype(IM_LABEL_RATIO);
                        Dtype y_dst = Dtype(i-4)/Dtype(IM_LABEL_RATIO);
                        Dtype depth_val_old = *(top[bottom.size()+sparse_processed-1]->cpu_data() + top[bottom.size()+sparse_processed-1]->offset(n,0,(int)y_dst) + (int)x_dst); // assumes top[bottom.size()] stores depths_sparse_labels
                        if (depth_val_old == 0 || depth_val < depth_val_old)
                           *(top[bottom.size()+sparse_processed-1]->mutable_cpu_data() + top[bottom.size()+sparse_processed-1]->offset(n,0,(int)y_dst) + (int)x_dst) = depth_val;          
                     }
                     *(top_pointer + top[index]->offset(n,0,i) + j) = depth_val;
                 }
             }                                         
          } // end of depths processing


	  /// start of normals processing
          if ((param_.data_type(index) == DataAugmentationParameter_DataType_NORMALS_LABELS)) 
          {
             cv::Mat bottom_mat, bottom_resized_mat, top_mat;
             bottom_mat = cv::Mat(height_[index], width_[index], CV_32FC3);
             float *bottom_mat_pointer = (float*) bottom_mat.data;
             bottom_resized_mat = cv::Mat(resized_height_, resized_width_, CV_32FC3);
             top_mat = bottom_resized_mat.colRange(crop_dims[0], crop_dims[1]).rowRange(crop_dims[2], crop_dims[3]);

             int ii = 0;   
    
             for (int i=0; i < height_[index]; ++i) 
             {
                 for (int j = 0; j < width_[index]; ++j) 
                 {
                     int j_;
                     if (f_) 
                        j_ = width_[index] - j - 1;
                     else    
                        j_ = j;  	

                     bottom_mat_pointer[ii    ] = *(bottom_pointer + bottom[index]->offset(n,0,i) + j_);
		     bottom_mat_pointer[ii + 1] = *(bottom_pointer + bottom[index]->offset(n,1,i) + j_);
		     bottom_mat_pointer[ii + 2] = *(bottom_pointer + bottom[index]->offset(n,2,i) + j_);
                     ii+=3;
                 }
             }
                        
             if (r_ != Dtype(0)) 
             {
                cv::Point2f center((Dtype)resized_height_/2., (Dtype)resized_width_/2.);
                Dtype alpha_ = cos(Dtype(-1)*r_* PI / 180.0), beta_ = sin(Dtype(-1)*r_* PI / 180.0);
                rot_coeff_[0] = alpha_;
                rot_coeff_[1] = beta_;
                rot_coeff_[2] = (1-alpha_)*center.x - beta_*center.y;
                rot_coeff_[3] = -beta_;
                rot_coeff_[4] = alpha_;
                rot_coeff_[5] = beta_*center.x - (1-alpha_)*center.y;
             }
               
             ii = 0;

             float *bottom_resized_mat_pointer = (float*) bottom_resized_mat.data;
             for (int i = 0; i < resized_height_; ++i) 
             {
                 for (int j = 0; j < resized_width_; ++j) 
                 { 
                     Dtype x_src, y_src;
                     if (r_ != Dtype(0)) 
                     {
		        x_src = (rot_coeff_[0]*Dtype(j) + rot_coeff_[1]*Dtype(i) + rot_coeff_[2])/s_;
                        y_src = (rot_coeff_[3]*Dtype(j) + rot_coeff_[4]*Dtype(i) + rot_coeff_[5])/s_;
                     }
                     else 
                     {
                        x_src = (Dtype) j/s_;
                        y_src = (Dtype) i/s_;
                     }

                     if (x_src < 0 || x_src >= width_[index] || y_src < 0 || y_src >= height_[index]) 
                     { 
                        bottom_resized_mat_pointer[ii    ] = 0;
		        bottom_resized_mat_pointer[ii + 1] = 0;
		        bottom_resized_mat_pointer[ii + 2] = 0;
                     }
                     else 
                     {
                        bottom_resized_mat_pointer[ii    ] = bottom_mat.at<cv::Vec3f>((int) y_src, (int) x_src)[0];
		        bottom_resized_mat_pointer[ii + 1] = bottom_mat.at<cv::Vec3f>((int) y_src, (int) x_src)[1];
		        bottom_resized_mat_pointer[ii + 2] = bottom_mat.at<cv::Vec3f>((int) y_src, (int) x_src)[2]; 
                     }                 
                     ii+=3;    
                 }
             }

             for (int i = 0; i < crop_height_[index]; ++i) 
             {
                 for (int j = 0; j < crop_width_[index]; ++j) 
                 {
		     Dtype x_val = top_mat.at<cv::Vec3f>(i,j)[0];
		     Dtype y_val = top_mat.at<cv::Vec3f>(i,j)[1];
		     Dtype z_val = top_mat.at<cv::Vec3f>(i,j)[2];
				
                     if (f_)
                        x_val = -x_val;

                     if (r_ != Dtype(0))
                     {
                        x_val = x_val*cos(Dtype(-1)*r_* PI / 180.0) - y_val*sin(Dtype(-1)*r_* PI / 180.0);
                        y_val = x_val*sin(Dtype(-1)*r_* PI / 180.0) + y_val*cos(Dtype(-1)*r_* PI / 180.0);
                     } 

                     if (s_ != Dtype(1))
                     {
                        z_val = z_val*s_;
                        Dtype norm_ = sqrt(x_val*x_val + y_val*y_val + z_val*z_val);
                        if (norm_ == 0)
                           norm_ = 1;
                        x_val = x_val/norm_;
                        y_val = y_val/norm_;
                        z_val = z_val/norm_;
                     }

                     *(top_pointer + top[index]->offset(n,0,i) + j) = x_val;
                     *(top_pointer + top[index]->offset(n,1,i) + j) = y_val;
                     *(top_pointer + top[index]->offset(n,2,i) + j) = z_val;
                 }
             }
                      
          } // end of normals processing
     
      } // index
    } // n
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(DataAugmentationLayer);
#endif

INSTANTIATE_CLASS(DataAugmentationLayer);
REGISTER_LAYER_CLASS(DataAugmentation);
}  // namespace caffe

