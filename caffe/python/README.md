## Details about implementing the layer

The layer is replication of the GVNN layer written in Torch by Handa et al. The layer has been written by Huangying Zhan and has been tweaked by me. 

PUT IN THE DESCRIPTION OF THE SUB LAYERS! 


The prototxt to use with the layer is as follows: 

Prototxt for using the **SE3 Generator layer**: 

'''
layer {
  type: "Python"
  name: "get_4x4_transformation_matrix_from_6dof"
  top: "get_4x4_transformation_matrix_from_6_dof"
  bottom: "bottom" // the blob that has your 6 DoF's 
	
  python_param {
    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    module: "pygeometry"
    # the layer name -- the class name in the module
    layer: "SE3_Generator"
    param_str : '{"threshold": 1e-12}' #I just gave a value need to check HELP 
  }
}
''' 

The prototxt for the **3D Grid Generator**:

'''
layer {
  type: "Python"
  name: "get_the_3D_points"
  top: "get_the_3D_points"
 
  bottom: "cropped_depth"
  bottom: "get_4x4_transformation_matrix" 	
	  
  python_param {
    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    module: "pygeometry"
    # the layer name -- the class name in the module
    layer: "Transform3DGrid"
    param_str : '{"fx":  259.425 ,"fy" : 259.73 , "cx" : 116.3 , "cy": 120.36 }' #put in suitable values HELP
  }
}
'''

The prototxt for the **Projection Layer**:

'''
layer {
  type: "Python"
  name: "get_the_flow"
  top: "get_the_flow"
  bottom: "get_the_3D_points" 
	
  python_param {
    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    module: 'pygeometry'
    # the layer name -- the class name in the module
    layer: 'PinHoleCamProj'
    param_str:'{"fx":  259.425 ,"fy" : 259.73 , "cx" : 116.3 , "cy": 120.36,"grid_normalized": False , "flowFlag": True}' #Need to check whether the values assigned are correct ? HELP 
  }
}
'''
