import caffe
import numpy as np

class SE3_Generator(caffe.Layer):
    """
    SE3_Generator takes 6 transformation parameters (se3) and generate corresponding 4x4 transformation matrix
    Input: 
        bottom[0] | se3 | shape is (batchsize, 6)
    Output: 
        top[0]    | SE3 | shape is (batchsize, 1, 4, 4)
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one input to compute transformation matrix.")

        # Define variables
        self.batchsize = bottom[0].num
        params = eval(self.param_str)
        self.threshold = params['threshold']

    def reshape(self, bottom, top):
        # check input dimension
        if bottom[0].count%6 != 0: #bottom.shape = (batchsize,6)
            raise Exception("Inputs must have the correct dimension.")
        #bottom[0].reshape(bottom[0].num,6)
	# Output is 4x4 transformation matrix
        top[0].reshape(bottom[0].num,1,4,4)

    def forward(self, bottom, top):
        # start = time.time() 
        # Define skew matrix of so3, .size = (batchsize,1,3,3)
        self.uw = bottom[0].data[:,:3]
        self.uw_x = np.zeros((self.batchsize,1,3,3))
        self.uw_x[:,0,0,1] = -self.uw[:,2]
        self.uw_x[:,0,0,2] = self.uw[:,1]
        self.uw_x[:,0,1,0] = self.uw[:,2]
        self.uw_x[:,0,1,2] = -self.uw[:,0]
        self.uw_x[:,0,2,0] = -self.uw[:,1]
        self.uw_x[:,0,2,1] = self.uw[:,0]

        # Get translation lie algebra
        self.ut = bottom[0].data[:,3:]
        self.ut = np.reshape(self.ut, (self.batchsize,1,3,1))

        # Calculate SO3 and T, i.e. rotation matrix (batchsize,1,3,3) and translation matrix (batchsize,1,1,3)
        self.R = np.zeros((self.batchsize,1,3,3))
        # self.V = np.zeros((self.batchsize,1,3,3))
        self.R[:,0] = np.eye(3)
        # self.V[:,0] = np.eye(3)
        self.theta = np.linalg.norm(self.uw,axis=1) #theta.size = (batchsize,1)
        # c1 = np.sin(self.theta)/self.theta
        # c2 = 2*np.sin(self.theta/2)**2/self.theta**2
        # c3 = ((self.theta - np.sin(self.theta))/self.theta**3)**2
        for i in range(self.batchsize):
            if self.theta[i]**2 < self.threshold:
                self.R[i,0] += self.uw_x[i,0]
                # self.V[i,0] += 0.5 * self.uw_x[i,0]
                continue
            else:
                c1 = np.sin(self.theta[i])/self.theta[i]
                c2 = 2*np.sin(self.theta[i]/2)**2/self.theta[i]**2
                c3 = ((self.theta[i] - np.sin(self.theta[i]))/self.theta[i]**3)**2
                self.R[i,0] += c1*self.uw_x[i,0] + c2*np.dot(self.uw_x[i,0],self.uw_x[i,0])
                # self.V[i,0] += c2*self.uw_x[i,0] + c3*np.dot(self.uw_x[i,0],self.uw_x[i,0])
        
        # Calculate output
        top[0].data[:,:,:3,:3] = self.R
        # top[0].data[:,:,:3,3] = np.matmul(self.V, self.ut)[:,:,:,0]
        # gvnn implementation
        top[0].data[:,:,:3,3] = np.matmul(self.R, self.ut)[:,:,:,0]
        top[0].data[:,:,3,3] = 1

        # debug
        # end = time.time()
        # print "SE3 generator forward: ", end-start

    # def dRduw_i(self, index):
    #     I3 = np.zeros((self.batchsize,1,3,3))
    #     I3[:,0] = np.eye(3)
    #     ei = np.zeros((self.batchsize,1,3,1))
    #     ei[:,0,index] = 1
    #     cross_term = np.matmul(self.uw_x, np.matmul(I3-self.R,ei))
    #     cross = np.zeros((self.batchsize,1,3,3))
    #     cross[:,0,0,1] = -cross_term[:,0,2]
    #     cross[:,0,0,2] = cross_term[:,0,1]
    #     cross[:,0,1,0] = cross_term[:,0,2]
    #     cross[:,0,1,2] = -cross_term[:,0,0]
    #     cross[:,0,2,0] = -cross_term[:,0,1]
    #     cross[:,0,2,1] = cross_term[:,0,0]
    #     self.dRduw_i = np.zeros((self.batchsize,1,3,3))
    #     for j in range(self.batchsize):
    #         if self.theta[j]**2 < self.threshold:
    #             self.dRduw_i[j] = self.uw_x[j]
    #         else:
    #             self.dRduw_i[j,0] = np.matmul((self.uw[j,index]*self.uw_x[j,0] + cross[j,0])/(self.theta[j]**2),self.R[j,0])

    def backward(self, top, propagate_down, bottom):
        # start = time.time()
        if propagate_down[0]:
            # top[0].diff .shape is (batchsize,1,4,4)
            # dLdut
            dLdT = top[0].diff[:,:,:3,3].reshape(self.batchsize,1,1,3)
            # gvnn implementation for DLdut is dLdT x R
            # dLdut = np.matmul(dLdT, self.V)
            dLdut = np.matmul(dLdT, self.R)
            bottom[0].diff[:,3:] = dLdut[:,0,0]
            # Gradient correction for dLdR. '.' R also affect T, need update dLdR
            grad_corr = np.matmul(np.swapaxes(dLdT, 2, 3), np.swapaxes(self.ut, 2, 3))  # from (b,hw,4,1) to (b,4,hw,1)



            # dLduw
            dLdR = top[0].diff[:,:,:3,:3]
            dLdR += grad_corr
            dLduw = np.zeros((self.batchsize,3))
            # for theta less than threshold
            generators = np.zeros((3,3,3))
            generators[0] = np.array([[0,0,0],[0,0,1],[0,-1,0]])
            generators[1] = np.array([[0,0,-1],[0,0,0],[1,0,0]])
            generators[2] = np.array([[0,1,0],[-1,0,0],[0,0,0]])
            for index in range(3):
                I3 = np.zeros((self.batchsize,1,3,3))
                I3[:,0] = np.eye(3)
                ei = np.zeros((self.batchsize,1,3,1))
                ei[:,0,index] = 1
                cross_term = np.matmul(self.uw_x, np.matmul(I3-self.R,ei))
                cross = np.zeros((self.batchsize,1,3,3))
                cross[:,0,0,1] = -cross_term[:,0,2,0]
                cross[:,0,0,2] = cross_term[:,0,1,0]
                cross[:,0,1,0] = cross_term[:,0,2,0]
                cross[:,0,1,2] = -cross_term[:,0,0,0]
                cross[:,0,2,0] = -cross_term[:,0,1,0]
                cross[:,0,2,1] = cross_term[:,0,0,0]
                self.dRduw_i = np.zeros((self.batchsize,1,3,3))
                for j in range(self.batchsize):
                    if self.theta[j]**2 < self.threshold:
                        self.dRduw_i[j] = generators[index]
                    else:
                        self.dRduw_i[j,0] = np.matmul((self.uw[j,index]*self.uw_x[j,0] + cross[j,0])/(self.theta[j]**2), self.R[j,0])
                dLduw[:,index]=np.sum(np.sum(dLdR*self.dRduw_i,axis=2),axis=2)[:,0]
            bottom[0].diff[:,:3] = dLduw



class Transform3DGrid(caffe.Layer):
    """
    Transform3DGrid takes Depth map to generate a 3D grid/3D points; then transform the grid/points according to transformation matrix
    Input: 
        Bottom[0] | Depth map | shape is (batchsize, 1, Height, Width)
        Bottom[1] | SE3       | shape is (batchsize, 1, 4, 4)
    Output:
        top[0]    | Transformed 3D points | shape is (batchsize, 4, Height, Width)

    """
    def setup(self, bottom, top):
        # check input pair 
        # bottom[0] is depth, shape is (batchsize, 1, height, width), second channel is depth
        # bottom[1] is 4x4 transformation matrix), shape is (batchsize, 1, 4, 4)
        if len(bottom) != 2:
            raise Exception("Need two inputs (Depth map & Transformation matrix) to generate 3D points/grid and transform it.")
        # Define variables
        self.batchsize = bottom[0].num
        # Layer parameters (Hardcode now, but to be edited later)
        params = eval(self.param_str)
        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']

    def reshape(self, bottom, top):
        # Check transformation matrix
        if bottom[1].channels != 1 or bottom[1].height != 4 or bottom[1].width != 4: 
            raise Exception("Bottom[1] must have the correct dimension: (Batchsize, 1, 4, 4)")
        # if bottom[0]. != 4:
            # raise Exception("Bottom[0] must have the correct dimension: (Batchsize, 1, Height, Width)")
        # top[0] has shape (batchsize, 4, height, width)
        top[0].reshape(self.batchsize, 4, bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        # Initialize grid
        self.image_size = [bottom[0].height, bottom[0].width]
        self.hw = self.image_size[0] * self.image_size[1]
        self.grid = np.ones((self.batchsize, self.hw, 4, 1)) 
        self.Z = bottom[0].data.reshape(self.batchsize, self.hw)
        # FIXME prevent depth = 0
        self.Z += 1e-9
        self.X = np.zeros(self.Z.shape)
        self.X[:] = (np.arange(self.hw) % self.image_size[1])+1
        self.Y = np.zeros(self.Z.shape)
        # self.Y[:] = (np.arange(self.hw) / self.image_size[0])+1
        self.Y[:] = (np.arange(self.hw) / self.image_size[1])+1
        # Calculate values in the grid
        self.X = (self.X  - self.cx) / self.fx
        self.Y = (self.Y  - self.cy) / self.fy 
        self.grid[:,:,0,0] = self.X * self.Z
        self.grid[:,:,1,0] = self.Y * self.Z
        self.grid[:,:,2,0] = self.Z
	
        # Transform points
        self.SE3 = bottom[1].data[...]
        self.grid_t = np.matmul(self.SE3, self.grid)
        self.grid_t = np.swapaxes(self.grid_t, 1, 2) # from (b,hw,4,1) to (b,4,hw,1)
        self.grid_t = self.grid_t.reshape(self.batchsize,4,self.image_size[0],self.image_size[1]) # from (b,4,hw,1) to (b,4,h,w)
        top[0].data[...] = self.grid_t

    def backward(self, top, propagate_down, bottom):
        dLdgrid_t = top[0].diff #shape is (b, 4, h, w)
        if propagate_down[1]:
            # dLdSE3, gradient of transformation matrix
            dLdgrid_t = dLdgrid_t.reshape(self.batchsize,4,self.hw) # reshape from (b,4,h,w) to (b,4,hw)
            self.grid = self.grid.reshape(self.grid.shape[:3]) # reshape from (b,hw,4,1) to (b,hw,4)
            dLdM = np.matmul(dLdgrid_t,self.grid)
            bottom[1].diff[...] = dLdM.reshape(self.batchsize,1,4,4)

        if propagate_down[0]:
            # dLdD, gradient of Depth
            baseGrid = np.zeros((self.batchsize, 1, 4, self.hw))
            baseGrid[:,0,2] = 1
            baseGrid[:,0,0] = self.X
            baseGrid[:,0,1] = self.Y
            y1 = np.matmul(self.SE3, baseGrid) # shape (b,1,4,hw)
            x1 = dLdgrid_t.reshape(self.batchsize,1,4,self.hw) # reshape from (b,4,hw) to (b,1,4,hw)
            bottom[0].diff[...] = np.sum(x1*y1,axis=2).reshape(self.batchsize, 1, self.image_size[0], self.image_size[1])


class PinHoleCamProj(caffe.Layer):
    """
    PinHoleCamProj takes 3D grid/points and back project thest points onto 2D image plane
    Input: 
        bottom[0] | 3D points | shape is (batchsize, 4, height, width), second channel has [X,Y,Z,1]
    Output: 
        top[0]    | 2D points | shape is (batchsize, 2, height, width), second channel are [u,v] on image plane. positive is to right/down
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one input to compute transformation matrix.")

        # Define variables
        self.batchsize = bottom[0].num
        self.image_size = [bottom[0].height, bottom[0].width]

        # Normalized grid setup
        # Layer parameters (Hardcode now, but to be edited later)
        params = eval(self.param_str)
        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']
        self.normFlag = params['grid_normalized']
        if self.normFlag:
            self.cx = (self.cx-1.0)/(self.image_size[1] - 1) * 2.0 - 1
            self.cy = (self.cy-1.0)/(self.image_size[0] - 1) * 2.0 - 1
            self.fx = (self.fx)/(self.image_size[1] - 1.0) * 2.0
            self.fy = (self.fy)/(self.image_size[0] - 1.0) * 2.0
        self.flowFlag  = params['flowFlag'] # if true, top[0] is the horizontal/vertical flow of target grid; else, top[0] is the target grid coordinates on source frame

    def reshape(self, bottom, top):
        # Output is horizontal and vertical flows.
        top[0].reshape(bottom[0].num, 2, self.image_size[0], self.image_size[1])


    def forward(self, bottom, top):
        # Vertical coordinate of projected grid
        top[0].data[:,1] = self.fy * bottom[0].data[:,1] / bottom[0].data[:,2] + self.cy -1
        # Horizontal coordinate of projected grid
    	top[0].data[:,0] = self.fx * bottom[0].data[:,0] / bottom[0].data[:,2] + self.cx -1
        if self.flowFlag:
            if self.normFlag:
                tmpArr = np.zeros(top[0].data.shape)
                tmpArr[:,1] = np.repeat(np.arange(self.image_size[0]).reshape(self.image_size[0],1),self.image_size[1],axis=1) # vertical
                tmpArr[:,1] = tmpArr[:,1]/(self.image_size[0]-1)*2-1
                tmpArr[:,0] = np.repeat(np.arange(self.image_size[1]).reshape(1, self.image_size[1]),self.image_size[0],axis=0) #horizontal
                tmpArr[:,0] = tmpArr[:,0]/(self.image_size[1]-1)*2-1
                top[0].data[...] -= tmpArr
            else:
                tmpArr = np.zeros(top[0].data.shape)
                tmpArr[:,1] = np.repeat(np.arange(self.image_size[0]).reshape(self.image_size[0],1),self.image_size[1],axis=1)
                tmpArr[:,0] = np.repeat(np.arange(self.image_size[1]).reshape(1, self.image_size[1]),self.image_size[0],axis=0)
                top[0].data[...] -= tmpArr
            
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
        	bottom[0].diff[:,0] = top[0].diff[:,0]*self.fx / bottom[0].data[:,2]
        	bottom[0].diff[:,1] = top[0].diff[:,1]*self.fy / bottom[0].data[:,2]
        	bottom[0].diff[:,2] = bottom[0].diff[:,0] * (-bottom[0].data[:,0]/bottom[0].data[:,2]) + bottom[0].diff[:,1] * (-bottom[0].data[:,1]/bottom[0].data[:,2]) 
