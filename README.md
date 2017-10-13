# Depth and pose estimation using ConvNets

The idea is to create a full fledged SLAM system using DL(Deep Learning). For the same we have considered the 'baby steps' which are as follows: 

- getting the pose in an unsupervised manner. (Trying to improve the results! Any suggestions feel free to PR)
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

# Pre Requisites (System Requirements)

The following need to be installed:

- Anaconda installation with all the latest dependencies which includes openCV and many othr libraries that CAFFE uses at the back 
- a C++ Compiler.
- CUDA 7.5 or CUDA 8.0 libraries along with a GPU. The depth network is pretty huge so you need to have enough GPU space to train the network.
- [CAFFE(Convolutional Architecture for Fast Feature Extraction)](http://caffe.berkeleyvision.org/)
  Please follow the instructions given on the website and build CAFFE with Python Layer Support as three layers that we would be using are written in python. Also add the layers that has been included in the caffe directory in the repository at suitable locations in your system so that your caffe has all the layers and mathematical tools to run our network.

That's pretty much all you need to get started! 
  
# Installations

The details of installtions of all the above are mentioned on their offcial wesite. Should not be that difficult. **In order to use our layers you need to change the .proto file. We have included our .proto file in the reposoitory for reference.**

The following tutorial might help: 
> [Simple-Example:-Sin-Layer](https://github.com/BVLC/caffe/wiki/Simple-Example:-Sin-Layer)
> [Making a layer in caffe](https://chrischoy.github.io/research/making-caffe-layer/)

This will help you in understanding ow to deploy a simple caffe custom layer from scratch!  

# Training the Network

Once though with all the installations and the dependencies, we come to train our network. For the same first we need to build the dataset. A script for the same has been included, build_dataset.py




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





