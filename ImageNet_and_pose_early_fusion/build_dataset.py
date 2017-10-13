#!/usr/bin/env python2.7
import sys

tools_path = "./tools"
sys.path.insert(0, tools_path)

from dataset_builder import NYUv2_loader
from dataset_builder import SceneNet_loader
# import dataset_builder

dataset = "NYU"
phase = "train"
datasetID = "0"
frame_distance = 5
pair_num = 1

if dataset == "NYU":
	if phase == "train":
		NYUv2_opt = {}
		NYUv2_opt["directory"] = "./miniRaw_1"
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
		NYUv2_opt["directory"] = "./miniRaw_1"
		NYUv2_opt["frame_distance"] = frame_distance
		NYUv2_opt["pair_num"] = pair_num
		NYUv2_opt["phase"] = "Test"
		NYUv2_opt["src_txt_path"] = "./resource/NYUv2_" + datasetID + "/Isrc_val.txt"
		NYUv2_opt["tgt_txt_path"] = "./resource/NYUv2_" + datasetID + "/Itgt_val.txt"
		NYUv2_opt["camMotion_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/camMotion_lmdb_val"
		NYUv2_opt["six_dof_lmdb_path"] = "./resource/NYUv2_" + datasetID + "/six_dof_lmdb_val"		
		nyu = NYUv2_loader(NYUv2_opt)
		nyu.main()
elif dataset == "SceneNet":
	if phase == "train":
		SceneNet_opt = {}
		SceneNet_opt["directory"] = "./SceneNet"
		SceneNet_opt["frame_distance"] = frame_distance
		SceneNet_opt["pair_num"] = pair_num
		SceneNet_opt["phase"] = "train"
		SceneNet_opt["src_txt_path"] = "./resource/SceneNet_" + datasetID + "/Isrc_train.txt"
		SceneNet_opt["tgt_txt_path"] = "./resource/SceneNet_" + datasetID + "/Itgt_train.txt"
		SceneNet_opt["camMotion_lmdb_path"] = "./resource/SceneNet_" + datasetID + "/camMotion_lmdb_train"
		SceneNet = SceneNet_loader(SceneNet_opt)
		SceneNet.main()
	elif phase == "val":
		SceneNet_opt = {}
		SceneNet_opt["directory"] = "./SceneNet"
		SceneNet_opt["frame_distance"] = frame_distance
		SceneNet_opt["pair_num"] = pair_num
		SceneNet_opt["phase"] = "val"
		SceneNet_opt["src_txt_path"] = "./resource/SceneNet_" + datasetID + "/Isrc_val.txt"
		SceneNet_opt["tgt_txt_path"] = "./resource/SceneNet_" + datasetID + "/Itgt_val.txt"
		SceneNet_opt["camMotion_lmdb_path"] = "./resource/SceneNet_" + datasetID + "/camMotion_lmdb_val"
		SceneNet = SceneNet_loader(SceneNet_opt)
		SceneNet.main()

