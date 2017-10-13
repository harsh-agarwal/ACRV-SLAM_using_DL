# Depth and Pose Estimation using ConvNets
The idea is to create a full-fledged vSLAM system using DL(Deep Learning). For the same we have considered the 'baby steps' which are as follows: 

- getting the pose in an unsupervised manner using photmetric loss. (Trying to improve the results! Any suggestions feel free to PR)
- checking the idea if depth features can be used to get a pose! 
- designing a depth network.
- working towards merging all the above elements. (TO BE DONE! Still Working on it)

As a student I would like to mention that in order to understand the approach and the code, I think the following pre-requisites are needed: 

- basic understanding of DL and Neural Networks 
  * ConvNets
  * Back Propagation 
  * Ideas behind various architectures (primarily ResNet and AlexNet)
  * Regularizers 
- Fundamental cincepts in Computer Vision 
  * Optic Flow 
  * SE3 and Lie Algebra (Basics)
  * Basic projection equations in vision 
- languages 
  * python
  * C++
  * [CAFFE(Convolutional Architecture for Fast Feature Extraction)](http://caffe.berkeleyvision.org/)(framework)

I have prepared a log that I followed before starting the project you might find it useful to start with. [A begginer's guide to vision and DL](https://harsh-agarwal.github.io/A-start!/)

# Pre-requisites (System Requirements)

The following libararies and softwares need to be installed:

- [Anaconda](https://conda.io/docs/install/quick.html) installation with all the latest dependencies which includes openCV and many othr libraries that CAFFE uses at the back 
- a C++ Compiler(msut be there! Just do version check!)
- CUDA 7.5 or CUDA 8.0 libraries along with a GPU. The depth network is pretty huge so you need to have enough GPU space to train the network.
- [CAFFE(Convolutional Architecture for Fast Feature Extraction)](http://caffe.berkeleyvision.org/)
  Please follow the instructions given on the website and build CAFFE with Python Layer Support as three layers that we would be using are written in python. Also add the layers that has been included in the caffe directory in the repository at suitable locations in your system so that your caffe has all the layers and mathematical tools to run our network.

That's pretty much all you need to get started! 
  
# Installations

The details of installtions of all the above are mentioned on their offcial wesite. Should not be that difficult. **In order to use our layers you need to change the .proto file. We have included our .proto file in the reposoitory for reference.**

The following tutorial might help: 
> [Simple-Example:-Sin-Layer](https://github.com/BVLC/caffe/wiki/Simple-Example:-Sin-Layer)
[Making a layer in caffe](https://chrischoy.github.io/research/making-caffe-layer/)

This will help you in understanding ow to deploy a simple caffe custom layer from scratch!  

# Details of custom layers  

## Hunagying's Layer 
We created three layers in python (look into _pygeometry.py_), the details of which are as follows. These layers are inline to the implementation by [GVNN implementation by Handa et al. in Torch](https://github.com/ankurhanda/gvnn): 

- **SE3 Generator Layer:** This layer would be used for converting the 6 DoF’s that we predict for getting the 4x4 Transformation Matrix 
- **Transform 3D Grid:** You need to transform the Target Depth according to the estimated pose.  
- **Pin Hole Camera Projection:** You need to project the transformed depth onto the image, or equivalently get the flow and add this flow to your source image to obtain your target image. 

The mathematics behind it might seem very complicated. No worries, [Huangying Zhan](https://www.roboticvision.org/rv_person/huangying-zhan/) had prepared a nice documentation regarding the derivation of the formula. The complete documentation can be found here. [Spatial Transformer Network Formulation by Huangying Zhan](https://drive.google.com/file/d/0B3BMdiXdDUoKTzExVVctWHB2NTYzMTNROW85a0Jpa1ZybDNJ/view?usp=sharing) 

We would be referring them as "HuangYing's Layer" 

## Photometric Loss and Backwarp

Also the **Backwarp layer** and **Abs_loss_layer** has been used which was a part of ["Unsupervised CNN for single view depth estimation: Geometry to the rescue"] by Garg et al. at ACRV along with Prof. Ian Reid. 

## Layers in Depth Network 
For implemeting Eigen's network we reused and tweaked of some of the layers that were a part of "Dense monocular reconstruction using surface normals" by [Saroj Weerasekera](https://www.roboticvision.org/rv_person/saroj-weerasekera/). The description of the layer's used are as follows: 

- **sparse_depth_euclidean_pairwise_loss_layer** - Computes the Eigen's loss given $\log(depth)$ values.
- **sparse_log_layer**: Computes the $\log(non-zero)$ values.
- **data_augmentation_layer**: It can primarily making the following augmentations in order to prevent overfitting.
	* Crop
	* Scale
	* Colour 
	* Flip 
	* Rotate
	* Translate 
	  And all this happens with a probability that is specified in the layer parameters! 


# Training the Network

Once though with all the installations and the dependencies, we come to train our network. For the same first we need to build the dataset. A script for the same has been included, build_dataset.py. the following code snippet explains how to use the script.

```python
dataset = "NYU" # mention the name of the dataset use 
phase = "val" # which data train or validation is to be prepared
datasetID = "10" # which id as in so that we could create multiple times using different id's with varying parameters 
frame_distance = 25 # Pairs are [1,25], [2,26], [3,27] 
pair_num = 1 # how many pairs as in [1,25] [1,50] [1,75] if pair num = 3 and frame distance 25 

if dataset == "NYU":
	if phase == "train":
		NYUv2_opt = {}
		NYUv2_opt["directory"] = "./Raw" # base directory 
		NYUv2_opt["frame_distance"] = frame_distance
		NYUv2_opt["pair_num"] = pair_num
		NYUv2_opt["phase"] = "Train"
		NYUv2_opt["src_txt_path"] = "./resource/NYUv2_" + datasetID + "/Isrc_train.txt"
		NYUv2_opt["tgt_txt_path"] = "./resource/NYUv2_" + datasetID + "/Itgt_train.txt"
		NYUv2_opt["camMotion_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/camMotion_lmdb_train"
		NYUv2_opt["six_dof_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/six_dof_lmdb_train"
		nyu = NYUv2_loader(NYUv2_opt)
		nyu.main()
	elif phase == "val":
		NYUv2_opt = {}
		NYUv2_opt["directory"] = "./Raw"
		NYUv2_opt["frame_distance"] = frame_distance
		NYUv2_opt["pair_num"] = pair_num
		NYUv2_opt["phase"] = "Test"
		NYUv2_opt["src_txt_path"] = "./resource/NYUv2_" + datasetID + "/Isrc_val.txt"
		NYUv2_opt["tgt_txt_path"] = "./resource/NYUv2_" + datasetID + "/Itgt_val.txt"
		NYUv2_opt["camMotion_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/camMotion_lmdb_val"
		NYUv2_opt["six_dof_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/six_dof_lmdb_val"
		nyu = NYUv2_loader(NYUv2_opt)
		nyu.main()
 ```

Create a new directory: 

```
mkdir resource
cd resource 
mkdir NYUv2_(dataset_id)
```
and run the python script.
```
python build_dataset.py
```
The desired data should be created in the specified folder with the properties mentioned in the script! 

Caution: Do check the data once before you start training. 

Results:
 
Detailed results are under process. Till then you can have a qualitative idea about what are heading upto by looking at the [Using ImageNet to get Pose](https://harsh-agarwal.github.io/Can-imageNet-be-Used/) and [A U-Net Kind of Depth Net](https://harsh-agarwal.github.io/Depth-Network/)







