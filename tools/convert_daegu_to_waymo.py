import numpy as np
import os, pickle, glob, math
import open3d
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

path_root = '/home/cv11/OpenPCDet-master/data'
Scene_list = ['Songhae', 'Daegu_6']

class_map = {
        'car' : 'Vehicle',
        'truck' : 'Vehicle',
        'pedestrian' : 'Pedestrian',
        'motorcycle' : 'Cyclist',
        'bicycle' : 'Cyclist'
        }

Daegu_label = []
for Scene in Scene_list:
    print('Scene name : {}'.format(Scene))
    Daegu_pcd_folder = os.path.join(path_root, 'Daegu', Scene, 'LiDAR_raw')
    Daegu_label_folder = os.path.join(path_root, 'Daegu', Scene, 'LiDAR_gt')
    Processed_folder = os.path.join(path_root, 'Daegu_processed')
    Processed_scene_folder = os.path.join(path_root, 'Daegu_processed', Scene)
    if not os.path.isdir(Processed_scene_folder):
        os.makedirs(Processed_scene_folder)

    Daegu_labels = sorted(glob.glob(os.path.join(Daegu_label_folder, '*.xml')))
    scene_index_n = len(Daegu_labels)

    Daegu_scene_label = []
    for i in tqdm(range(scene_index_n)):
        #pcd -> npy format
        pcd_name = Path(Daegu_labels[i]).stem
        Daegu_pcd_path = os.path.join(Daegu_pcd_folder, pcd_name + '.pcd')
        pc = open3d.read_point_cloud(Daegu_pcd_path)
        pc_array = np.asarray(pc.points)
        pc_save_path = os.path.join(Processed_scene_folder, '{0:04d}'.format(i))

        cnt = 0
        # for i in range (len(len(pc_array))):
        #     if pc_array[i, :] != 0:
        #         new_pc_array[cnt, :] = pc_array[i, :]
        #         cnt += 1

        np.save(pc_save_path, pc_array)

        #Daegu xml -> Waymo pkl
        tree = ET.parse(Daegu_labels[i])
        root = tree.getroot()
        
        # generate annos (name, dimensions, location, obj_ids, num_points_in_gt)
        obj_ids = []
        name = []
        dimensions = []
        locations = []
        heading_angles = []
        num_points_in_gts = []
        tracking_difficulties = []
        gt_boxes_lidars = []

        for object in root.iter("object"):
            tracking_ids = object.find("bndbox").findtext("track_id")
            class_name_ori = object.find("class").findtext("sub_class")
            if class_name_ori == '""':
                class_name_ori = object.findtext("class").split("\n")[0]
                if class_name_ori != 'pedestrian' and class_name_ori != 'bicycle':
                    print(object.findtext("class"))
            class_name = class_map[class_name_ori]
            dimension = object.find("bndbox").findtext("dimension").split()
            location = object.find("bndbox").findtext("location").split()
            anchor_box = object.find("bndbox").findtext("box_anchor").split()

            num_points_in_gt = 1000
            tracking_difficulty = 0

            #width, length changed and pi/2

            # need angle calculation
            #heading_angle = -1*math.atan((float(anchor_box[12])-float(anchor_box[0])) / (float(anchor_box[13])-float(anchor_box[1])+0.00001))-(math.pi/2)
            # 1256 floor
            # 3478 upper
            # 1234 front
            # 5678 back
            np_anchor_box = np.array(anchor_box).astype(np.float32).reshape(8,3)
            # print(np_anchor_box)
            front_points = np_anchor_box[0:4,:]
            back_points = np_anchor_box[4:8,:]
            front_points_mean = front_points.mean(axis=0)
            back_points_mean = back_points.mean(axis=0)


            heading_angle = math.atan2((float(anchor_box[13])-float(anchor_box[1])), (float(anchor_box[12])-float(anchor_box[0])))
            heading_angle = math.atan2((front_points_mean[1] - back_points_mean[1]), (front_points_mean[0] - back_points_mean[0]))
            #dimension : waymo = length * width * height
            #            DAEGU = height * width * length [1]<->[2]
            # 2 1 0
            # 0 1 2
            # waymo = length * width * height
            height = dimension[0]
            width  = dimension[1]
            length = dimension[2]
            
            dimension = [float(length), float(width), float(height)]
            location = [float(location[0]), float(location[1]), float(location[2])]
            gt_boxes_lidar = [float(location[0]), float(location[1]), float(location[2]), float(dimension[0]), float(dimension[1]), float(dimension[2]), heading_angle]

            obj_ids.append(tracking_ids)
            name.append(class_name)
            dimensions.append(dimension)
            locations.append(location)
            heading_angles.append(heading_angle)
            num_points_in_gts.append(num_points_in_gt)
            tracking_difficulties.append(tracking_difficulty)
            gt_boxes_lidars.append(gt_boxes_lidar)

        obj_ids = np.asarray(obj_ids)
        name = np.asarray(name)
        dimension = np.asarray(dimension)
        locations = np.asarray(locations)
        heading_angles = np.asarray(heading_angles)
        num_points_in_gts = np.asarray(num_points_in_gts)
        tracking_difficulties = np.asarray(tracking_difficulties)
        gt_boxes_lidars = np.asarray(gt_boxes_lidars)

        Daegu_label.append({'point_cloud': {'num_features': 3, 'lidar_sequence': Scene, 'sample_idx': i},
        'frame_id': Scene + '_' + '{0:03d}'.format(i),
        'metadata': {'context_name': Scene + '_' + '{0:03d}'.format(i), 'timestamp_micros': 0},
        'image': {'image_shape_0': (1280, 1920), 'image_shape_1': (1280, 1920), 'image_shape_2': (1280,1920), 'image_shape_3': (1280, 1920), 'image_shape_4': (1280,1920)},
        'pose': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        'annos': {'name': name, 'difficulty': tracking_difficulties, 'dimensions': dimensions, 'location': locations, 'heading_angles': heading_angles, 'obj_ids': obj_ids,
        'tracking_difficulty': tracking_difficulties, 'num_points_in_gt': num_points_in_gts, 'gt_boxes_lidar': gt_boxes_lidars},
        'num_points_of_each_lidar': [len(pc_array), 0, 0, 0, 0]})


    #Scene pkl save
    with open(os.path.join(Processed_scene_folder, Scene + '.pkl'), 'wb') as f:
        pickle.dump(Daegu_label, f, pickle.HIGHEST_PROTOCOL)

#Daegu whole pkl save
with open(os.path.join(Processed_folder, 'Daegu_processed_data_infos_train.pkl'), 'wb') as f:
    pickle.dump(Daegu_label, f, pickle.HIGHEST_PROTOCOL)

# with open('/home/cv11/OpenPCDet-master/data/Daegu/waymo_processed_data_v0_5_0_infos_train.pkl', 'rb') as f:
#     waymo_label = pickle.load(f)

# with open('/home/cv11/OpenPCDet-master/data/Daegu/segment-15832924468527961_1564_160_1584_160_with_camera_labels.pkl', 'rb') as f:
#     waymo_scene_label = pickle.load(f)

# print(waymo_label[0])
# print(Daegu_label[0])
# print('complete')
