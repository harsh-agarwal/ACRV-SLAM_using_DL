#!/usr/bin/env python2.7
import os
import random
import scipy.io as sio
import numpy as np
import lmdb

import sys
from os.path import expanduser
home = expanduser("~")
caffe_root = home + '/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, './tools')
import caffe

#import sceneNetTools as sn_tool

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def save_txt(path, img_list):
    f = open(path, 'w')
    for line in img_list:
        f.writelines(line)
    f.close()

def readTxt(path):
    f = open(path, 'r')
    txt = f.readlines()
    f.close()
    return txt

def np2lmdb(path, np_arr):
    # input: np_arr: shape = (N,C,H,W)
    N = np_arr.shape[0]
    map_size = np_arr.nbytes * 10
    env = lmdb.open(path, map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = np_arr.shape[1]
            datum.height = np_arr.shape[2]
            datum.width = np_arr.shape[3]
            datum = caffe.io.array_to_datum(np_arr[i])
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


class NYUv2_loader():
    def __init__(self, NYUv2_opt):
        self.directory = NYUv2_opt["directory"]
        self.frame_distance = NYUv2_opt["frame_distance"]
        self.pair_num = NYUv2_opt["pair_num"]
        self.phase = NYUv2_opt["phase"]
        self.src_txt_path = NYUv2_opt["src_txt_path"]
        self.tgt_txt_path = NYUv2_opt["tgt_txt_path"]
        self.camMotion_lmdb_path = NYUv2_opt["camMotion_lmdb_path"]
        #adding path to be used for saving the 6DoF's as LMDB
        self.six_dof_lmdb_path = NYUv2_opt["six_dof_lmdb_path"]

        self.Isrc_list = []
        self.Itgt_list = []
        self.Dsrc_list = []
        self.Dtgt_list = []

        self.root_path = self.directory + "/" + self.phase
        assert os.path.isdir(self.root_path), '%s is not a valid root_path' % self.root_path
        self.sceneFolderList = os.listdir(self.root_path)
        self.sceneFolderList = ["/".join([self.root_path,i]) for i in self.sceneFolderList]
        self.sceneFolderList = [i for i in self.sceneFolderList if ".DS_Store" not in i]


    def sortImg(self,imgList):
        # sort image paths in numerical order
        # input: ["0.jpg", "1.jpg" ,"10.jpg", "2.jpg"]
        # output: ["0.jpg", "1.jpg", "2.jpg", "10.jpg"]
        affix = imgList[0].split(".")[-1]
        tmp = [i[:-4] for i in imgList]
        tmp.sort(key=int)
        return [".".join([i,affix]) for i in tmp]

    def getImages(self):
        # Get a list of list containing paths of images in different scenes
        # e.g. [[img_paths in scene 0],[img_paths in scene 1],[img_paths in scene 2]]
        self.images = []

        self.rgbFolderList = ["/".join([i,"rgb"]) for i in self.sceneFolderList]

        for folder in self.rgbFolderList:
            for root, _, fnames in sorted(os.walk(folder)):
                tmp = []
                fnames = sorted(fnames)
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root,fname)
                        tmp.append(path+"\n")
                self.images.append(tmp)

    def makeDatasetWithCamMotion(self):
        # Get three lists
        # 1. self.Isrc_list: source image paths
        # 2. self.Itgt_list: target image paths
        # 3. self.camMotionGT: transformation matrix from target frame to source frame
        self.camMotionGT = []
        self.six_dof = []
        for cnt in xrange(len(self.images)):
        # for folder in self.images:
            folder = self.images[cnt]
            scene = self.sceneFolderList[cnt]
            # load CamMotion
            cam_traj = sio.loadmat("/".join([scene,"CameraTrajectory.mat"]))["Twc"][0]
            end_line_cnt = len(folder) - self.frame_distance*self.pair_num
            for line_cnt in xrange(end_line_cnt):
            # for line_cnt, line in enumerate(folder[:end_line_cnt]):
                line = folder[line_cnt]
                srcArr = cam_traj[line_cnt]
                if srcArr.shape != (4,4):
                    continue
                for i in xrange(self.pair_num):
                    next_cnt = line_cnt + self.frame_distance * (i+1)
                    # camera Motion
                    tgtArr = cam_traj[next_cnt]
                    if tgtArr.shape != (4,4):
                        continue
                    final_transformation=np.dot(np.linalg.inv(srcArr), tgtArr)
                    self.camMotionGT.append(np.dot(np.linalg.inv(srcArr), tgtArr))
                    #converting to 6DoF's
                    R = final_transformation[:3,:3] 
                    tr_R = np.trace(R)
                    cos_theta = (tr_R - 1.0)/2.0
                    print cos_theta
                    print line
                    print folder[next_cnt]
                    theta = np.arccos(cos_theta)
                    sin_theta = np.sin(theta)
                    ln_R = (theta/(2*(sin_theta+1e-5)))*(R-np.transpose(R))
                    temp_6dof = np.zeros((6,1))
                    temp_6dof[2,0] = -ln_R[0,1]
                    temp_6dof[1,0] =  ln_R[0,2]
                    temp_6dof[0,0] = -ln_R[1,2] 
                    translation =np.dot(np.linalg.inv(R),final_transformation[0:3,3])                    
                    temp_6dof[3:,0] = translation
                    self.six_dof.append(np.transpose(temp_6dof))                    
                    # images
                    self.Isrc_list.append(line)
                    self.Itgt_list.append(folder[next_cnt])


    def shuffleDataset(self):
        # Shuffled the following three lists
        # 1. self.Isrc_list: source image paths
        # 2. self.Itgt_list: target image paths
        # 3. self.camMotionGT: transformation matrix from target frame to source frame
        Img_list = list(zip(self.Isrc_list, self.Itgt_list, self.camMotionGT,self.six_dof))
        random.shuffle(Img_list)
        self.Isrc_list, self.Itgt_list , self.camMotionGT , self.six_dof = zip(*Img_list)


    def camMotGT_list2arr(self):
        N = len(self.camMotionGT)
        self.camMotionGT = np.asarray(self.camMotionGT).reshape((N,1,4,4))
        self.six_dof = np.asarray(self.six_dof).reshape((N,6,1,1))

    def rgb2depth(self):
        self.Dsrc_list = [line.replace('rgb','depth') for line in self.Isrc_list]
        self.Dtgt_list = [line.replace('rgb','depth') for line in self.Itgt_list]

    def main(self):
        print("Getting images...")
        self.getImages()

        print("Making dataset...")
        self.makeDatasetWithCamMotion()

        print("Shuffling dataset...")
        self.shuffleDataset()

        print("Saving dataset...")
        save_txt(self.src_txt_path, self.Isrc_list)
        save_txt(self.tgt_txt_path, self.Itgt_list)
        self.camMotGT_list2arr()
        np2lmdb(self.camMotion_lmdb_path, self.camMotionGT)
        np2lmdb(self.six_dof_lmdb_path, self.six_dof)

        print("Saving depth...")
        self.rgb2depth()
        self.src_txt_path = self.src_txt_path.replace("Isrc","Dsrc")
        self.tgt_txt_path = self.tgt_txt_path.replace("Itgt","Dtgt")
        save_txt(self.src_txt_path, self.Dsrc_list)
        save_txt(self.tgt_txt_path, self.Dtgt_list)

class SceneNet_loader():
    def __init__(self, SceneNet_opt):
        self.directory = SceneNet_opt["directory"]
        self.frame_distance = SceneNet_opt["frame_distance"]
        self.pair_num = SceneNet_opt["pair_num"]
        self.phase = SceneNet_opt["phase"]
        self.src_txt_path = SceneNet_opt["src_txt_path"]
        self.tgt_txt_path = SceneNet_opt["tgt_txt_path"]
        self.camMotion_lmdb_path = SceneNet_opt["camMotion_lmdb_path"]
        self.Isrc_list = []
        self.Itgt_list = []
        self.Dsrc_list = []
        self.Dtgt_list = []
        self.images = {}
        self.root_path = self.directory + "/" + self.phase
        self.render_paths = []
        self.camMotion_Twc = {}
        self.camMotionGT = []

    def sortImg(self,imgList):
        affix = imgList[0].split(".")[-1]
        tmp = [i[:-4] for i in imgList]
        tmp.sort(key=int)
        return [".".join([i,affix]) for i in tmp]

    def getImages_old(self):
        assert os.path.isdir(self.root_path), '%s is not a valid root_path' % self.root_path

        s1 = os.listdir(self.root_path)
        sceneFolderList = []
        for i in s1:
            tmp = os.listdir("/".join([self.root_path, i]))
            tmp = ["/".join([self.root_path, i, j]) for j in tmp]
            sceneFolderList.append(tmp)

        for i in sceneFolderList:
            rgbFolderList = [j+"/photo" for j in i]

        for folder in rgbFolderList:
            for root, _, fnames in sorted(os.walk(folder)):
                tmp = []
                fnames = self.sortImg(fnames)
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root,fname)
                        tmp.append(path+"\n")
                self.images.append(tmp)

    def getImages(self):
        # save image paths as a dictionary
        # key: scene folder: e.g. 0/0
        # content: image paths in the folder
        assert os.path.isdir(self.root_path), '%s is not a valid root_path' % self.root_path  

        self.render_paths = []
        s1 = os.listdir(self.root_path)
        sceneFolderList = []
        for i in s1:
            tmp = os.listdir("/".join([self.root_path, i]))
            tmp = ["/".join([i, j]) for j in tmp]
            self.render_paths += tmp

        for render_path in self.render_paths:
            full_path = '/'.join([self.root_path, render_path, "photo"])
            for root, _, fnames in sorted(os.walk(full_path)):
                tmp = []
                fnames = self.sortImg(fnames)
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root,fname)
                        tmp.append(path+"\n")
                self.images[render_path] = tmp

    def getTwc(self):
        protobufs_folder = self.root_path + '_protobufs'
        protobufs = os.listdir(protobufs_folder)
        for protobuf in protobufs:
            protobuf_path = '/'.join([protobufs_folder, protobuf])
            trajectories = sn_tool.sn.Trajectories()
            try:
                with open(protobuf_path,'rb') as f:
                    trajectories.ParseFromString(f.read())
            except IOError:
                print('Scenenet protobuf data not found at location:{0}'.format(self.root_path))
                print('Please ensure you have copied the pb file to the data directory')

            for traj in trajectories.trajectories:
                Twc = np.zeros((len(traj.views),4,4))
                for idx,view in enumerate(traj.views):
                    # Get camera pose
                    ground_truth_pose = sn_tool.interpolate_poses(view.shutter_open,view.shutter_close,0.5)
                    Twc[idx] = sn_tool.camera_to_world_with_pose(ground_truth_pose)
                self.camMotion_Twc[traj.render_path] = Twc


    def makeDataset_old(self):
        for folder in self.images:
            end_line_cnt = len(folder) - self.frame_distance * self.pair_num
            for line_cnt, line in enumerate(folder[:end_line_cnt]):
                for i in xrange(self.pair_num):
                    next_cnt = line_cnt + self.frame_distance * (i+1)
                    self.Isrc_list.append(line)
                    self.Itgt_list.append(folder[next_cnt])

    def makeDatasetWithCamMotion(self):
        # Get three lists
        # 1. self.Isrc_list: source image paths
        # 2. self.Itgt_list: target image paths
        # 3. self.camMotionGT: transformation matrix from target frame to source frame
        for key in self.images:
            folder = self.images[key]
            end_line_cnt = len(folder) - self.frame_distance * self.pair_num
            cam_traj = self.camMotion_Twc[key]
            for line_cnt in xrange(end_line_cnt):   
                line = folder[line_cnt]
                srcArr = cam_traj[line_cnt]
                if srcArr.shape != (4,4):
                    continue
                for i in xrange(self.pair_num):
                    next_cnt = line_cnt + self.frame_distance * (i+1)
                    # camera Motion
                    tgtArr = cam_traj[next_cnt]
                    if tgtArr.shape != (4,4):
                        continue
                    self.camMotionGT.append(np.dot(np.linalg.inv(srcArr), tgtArr))
                    # images
                    self.Isrc_list.append(line)
                    self.Itgt_list.append(folder[next_cnt])

    def shuffleDataset(self):
        # Shuffled the following three lists
        # 1. self.Isrc_list: source image paths
        # 2. self.Itgt_list: target image paths
        # 3. self.camMotionGT: transformation matrix from target frame to source frame
        Img_list = list(zip(self.Isrc_list, self.Itgt_list, self.camMotionGT))
        random.shuffle(Img_list)
        self.Isrc_list, self.Itgt_list , self.camMotionGT= zip(*Img_list)


    def camMotGT_list2arr(self):
        N = len(self.camMotionGT)
        self.camMotionGT = np.asarray(self.camMotionGT).reshape((N,1,4,4))

    def rgb2depth(self):
        self.Dsrc_list = [line.replace('rgb','depth') for line in self.Isrc_list]
        self.Dtgt_list = [line.replace('rgb','depth') for line in self.Itgt_list]

    def main(self):
        print("Getting images...")
        self.getImages()

        print("Getting camera trajectories...")
        self.getTwc()

        print("Making dataset...")
        self.makeDatasetWithCamMotion()

        print("Shuffling dataset...")
        self.shuffleDataset()

        print("Saving dataset...")
        save_txt(self.src_txt_path, self.Isrc_list)
        save_txt(self.tgt_txt_path, self.Itgt_list)
        self.camMotGT_list2arr()
        np2lmdb(self.camMotion_lmdb_path, self.camMotionGT)

        print("Saving depth...")
        self.rgb2depth()
        self.src_txt_path = self.src_txt_path.replace("Isrc","Dsrc")
        self.tgt_txt_path = self.tgt_txt_path.replace("Itgt","Dtgt")
        save_txt(self.src_txt_path, self.Dsrc_list)
        save_txt(self.tgt_txt_path, self.Dtgt_list)

        # print("These files don't exist")
        # print self.notExistDepth



# if __name__ == "__main__" :
#     # phase = "Train"
#     # frame_distance = 2
#     # pair_num = 10
#     # NYU = NYUv2_loader(frame_distance, pair_num, phase)
#     # NYU.main()
#     # NYU.phase = "Test"
#     # NYU.main()

# SceneNet_opt = {}
# SceneNet_opt["directory"] = "./SceneNet"
# SceneNet_opt["frame_distance"] = 1
# SceneNet_opt["pair_num"] = 1
# SceneNet_opt["phase"] = "val"
# SceneNet_opt["src_txt_path"] = "./resource/SceneNet/Isrc_train_1.txt"
# SceneNet_opt["tgt_txt_path"] = "./resource/SceneNet/Itgt_train_1.txt"

# SceneNet = SceneNet_loader(SceneNet_opt)
#     SceneNet.main()


# NYUv2_opt = {}
# NYUv2_opt["directory"] = "./Raw"
# NYUv2_opt["frame_distance"] = 100
# NYUv2_opt["pair_num"] = 1
# NYUv2_opt["phase"] = "Train"
# NYUv2_opt["src_txt_path"] = "./resource/Isrc_train_2.txt"
# NYUv2_opt["tgt_txt_path"] = "./resource/Itgt_train_2.txt"
# nyu = NYUv2_loader(NYUv2_opt)
# nyu.getImages()

# nyu2 = NYUv2_loader(NYUv2_opt)
# nyu2.images = nyu.images
# nyu2.makeDatasetWithCamMotion()
