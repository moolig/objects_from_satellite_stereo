# -*- coding: utf-8 -*-

from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import math

import  detectron2.structures.boxes

from detectron2.data.datasets import register_coco_instances


def get_center(a, c):
    return c + (a-c)/2

def get_r(a, c):
    return np.linalg.norm(a-c)/2

def get_new_b(a, b, c):
    r = get_r(a, c)
    center = get_center(a, c)
    diraction = (b-center)/np.linalg.norm(b-center)
    new_point = center + r*diraction
    return new_point


def angle_clculate(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    print(np.degrees(angle))


    n_point = np.array([ba[1], -ba[0]], np.float64) * np.linalg.norm(bc) / np.linalg.norm(ba)
    # cosine_angle = np.dot(ba, n_point) / (np.linalg.norm(ba) * np.linalg.norm(n_point))
    # angle = np.arccos(cosine_angle)
    # print(np.degrees(angle))
    if(np.degrees(angle) > 150):
        return 1
    elif(np.degrees(angle) > 95 or np.degrees(angle) < 85):
        return 2
    else:
        return 3


    # if(np.degrees(angle) > 160):
    #     return b
    # else:
    #     new_b = get_new_b(a, b ,c)
    #     return new_b


def set_right_angles(polygon):
    s = polygon.shape[0]-1
    rmove_points = []
    fleg = True
    for i, point in enumerate(polygon):
        if(not fleg):
            fleg = True
            continue
        ret_val = angle_clculate(polygon[i], polygon[(i+1)%s], polygon[(i+2)%s])
        if(ret_val == 3):
            continue
        elif(ret_val == 2):
            fleg = False
            polygon[(i + 1)%s] = get_new_b(polygon[i], polygon[(i+1)%s], polygon[(i+2)%s])
        else:
            fleg = False
            rmove_points.append((i+1)%s)


        # polygon[(i + 1)%s] = ret_val

    polygon = np.delete(polygon, rmove_points, 0)
    return polygon


# root_air = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/AIRS//'
# train_image_root = root_air + r'/train/images'
# train_json_file = root_air + r'/train/instances_shape_train2018.json'
# val_image_root = root_air + r'/val/images'
# val_json_file = root_air + r'/val/coco_format.json'
#
# register_coco_instances("airs_dataset_train", {}, train_json_file, train_image_root)
# register_coco_instances("airs_dataset_val", {}, val_json_file, val_image_root)

if __name__ == "__main__":
    img_path = r'/home/shmuelgr/PycharmProjects/polyrnn-pp-pytorch/img/aerial.jpg'
    img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/AIRS/test/images/christchurch_407.tif'
    img_path = r'/home/shmuelgr/PycharmProjects/convert_dataset_type/pycococreator-master/examples/AerialInageDataset_test/val/images/austin11_w_0_h_1.jpg'
    # img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/smallAerialImageDataset/test/images/innsbruck15_w_4_h_8.jpg'
    img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/smallAerialImageDataset/test/images/innsbruck15_w_5_h_6.jpg'
    # img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/balloon/train/16435593892_2aa8118f4a_k.jpg'

    im = cv2.imread(img_path)
    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # ------------------------------------------------------------------------------------------------------------
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("airs_dataset_train",)
    cfg.DATASETS.TEST = ("airs_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    # ------------------------------------------------------------------------------------------------------------

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.WEIGHTS = r'/home/shmuelgr/PycharmProjects/detectron2/output/14_3_3x10^4_loop/model_final.pth'
    cfg.MODEL.WEIGHTS = r'/home/shmuelgr/PycharmProjects/detectron2/output/model_final.pth'
    # cfg.MODEL.WEIGHTS = r'/home/shmuelgr/PycharmProjects/detectron2/output/test2/model_final.pth'

    # cfg.MODEL.WEIGHTS = "/home/shmuelgr/PycharmProjects/detectron2/output/test2/model_final.pth"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs["instances"].pred_classes
    outputs["instances"].pred_boxes

    masks = (

        outputs["instances"]
            .get_fields()["pred_masks"]
            .to("cpu")
            .numpy()
    )

    h, w = im.shape[:2]
    all_masks = np.zeros((h, w), np.uint8)
    for i, mask in enumerate(masks):
        all_masks += mask

    all_masks = all_masks*255;
    _, binary = cv2.threshold(all_masks, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        cv2.drawContours(im, [approx], -1, (0, 255, 255), 1)



    for i, mask in enumerate(masks):
        print(i)
        # mask_im = Image.fromarray(mask)
        # mask_im.save("./output/mask" + str(i)+'.jpeg')
        a = np.uint8(mask)*255
        #
        out = cv2.Laplacian(a, cv2.CV_8U)
        # cv2.imwrite("./output/buord" + str(i)+'.jpeg', out)


        result = np.where(out == 255)
        x = result[1]
        y = result[0]

        a = np.array([x, y])
        b = a.T

        cent = (sum([p[0] for p in b]) / len(b), sum([p[1] for p in b]) / len(b))
        # sort by polar angle
        data_as_list = b.tolist()
        data_as_list.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
        # plot points
        poly1 = Polygon(data_as_list)
        x1, y1 = poly1.exterior.coords.xy
        poly2 = poly1.simplify(1)
        x2, y2 = poly2.exterior.coords.xy


        polygon = np.array([x2, y2]).T
        polygon2 = set_right_angles(polygon)
        x3 = polygon2.T[0]
        y3 = polygon2.T[1]

        # plt.figure(figsize=(9, 3))
        # plt.subplot(131)
        # plt.plot(y1, x1)
        # plt.subplot(132)
        # plt.plot(y2, x2)
        # plt.subplot(133)
        # plt.plot(y3, x3)
        # plt.suptitle('Categorical Plotting')
        # plt.show()

        polygon = np.array([x2, y2]).T.astype(int)
        for pix in polygon:
            im[pix[1], pix[0], 2] = 255

    crop = outputs["instances"].pred_boxes
    np_crop = crop.to('cpu').tensor.detach().numpy()
    np_crop_round = np.around(np_crop).astype(int)


    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    res_im = v.get_image()[:, :, ::-1]
    cv2.imwrite(r'/home/shmuelgr/PycharmProjects/detectron2/output/testres.tif', res_im)

    resized = cv2.resize(res_im, im.shape[:2])

    cv2.imshow('im', im)
    cv2.imshow('res_im', res_im)
    for box in np_crop_round:
        crop_im = im[box[1]:box[3], box[0]:box[2]]
        crop_res_im = resized[box[1]:box[3], box[0]:box[2]]

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5 * high_thresh

        edges = cv2.Canny(crop_im, 300, 500)
        cv2.imshow('crop_im', crop_im)
        cv2.imshow('crop_res_im', crop_res_im)
        cv2.imshow('edges', edges)
        cv2.waitKey(0)
    cv2.destroyAllWindows()




    plt.show()

    # df = create_csv(outputs["instances"])

