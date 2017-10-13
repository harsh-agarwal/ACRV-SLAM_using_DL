# ImageNet to get pose - Early Fusion

The primary idea to the project and the approach can be found in the second section of [Can ImageNet be used for gtting the pose](https://harsh-agarwal.github.io/Can-imageNet-be-Used/).

The new layers that we have written can be found in the README file of the Caffe folder along with the locations at which you need to place the respective files and build caffe!

Most of the current implementations use early fusion to get the pose. We wanted to qualitatively and quantitatively evaluate how much are we losing if we are going for late fusion. The blog has all the results! 

The PoseVsWarp directory has some basic network in order to sanity check our layers and get a graph between the warp error and pose error. The script was written just to ensure that the forward pass of the network, that is HuangYing's layer and the backwarp layer are working well! As expected the PoseError vs warpError graph comes out to be like: 

TODO: ADD THE GRAPHS! 

We are still working upon improving our late fusion model. Any suggestions are most welcome! 


