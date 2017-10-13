# U Net kind of Depth-Net

As we understood that depth features can be used to calculate the pose, our hypothesis was that given a good depth map of the target image and good depth features that can help us capture motion cues (or features analogous to that) we would be able to improve upon the pose that we get! This experiment was an attempt to design a depth network that can help us in accomplishing such features.

The details of the project and the way we wnt about it can be found in this blog [A U-Net kind of a Depth Net](https://harsh-agarwal.github.io/Depth-Network/). The architecture can be visualised using [Netscope](ethereon.github.io/netscope/#/editor). The network has been trained on NYUDv2 using Eigen's loss. We would be putting the quantitative results soon. 




