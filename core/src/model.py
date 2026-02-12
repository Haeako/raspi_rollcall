#!/usr/bin/env python3

import os
import cv2
import glob2
import sys

# import insightface
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage import transform
from tqdm import tqdm

sys.path.append("../../external/GhostFaceNet")
from face_detector import YoloV5FaceDetector

class Model:

    def __init__(self, model_file):
        self.det = YoloV5FaceDetector()

        if model_file is not None:
            self.face_model = tf.keras.models.load_model(model_file, compile=False)
        else:
            self.face_model = None

    def face_align_landmarks_sk(self, img, landmarks, image_size=(112,112), method="similar"):
        tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()

        src = np.array([
            [38.2946,51.6963],
            [73.5318,51.5014],
            [56.0252,71.7366],
            [41.5493,92.3655],
            [70.7299,92.2041]
        ], dtype=np.float32)

        ret = []
        for landmark in landmarks:
            tform.estimate(landmark, src)
            ret.append(transform.warp(img, tform.inverse, output_shape=image_size))

        return (np.array(ret) * 255).astype(np.uint8)

    def do_detect_in_image(self, image, image_format="BGR"):

        imm_BGR = image if image_format == "BGR" else image[:,:,::-1]
        imm_RGB = image[:,:,::-1] if image_format == "BGR" else image

        bboxes, pps, ccs = self.det(imm_BGR)

        nimgs = self.face_align_landmarks_sk(imm_RGB, pps)
        bbs = bboxes[:, :4].astype("int")
        ccs = bboxes[:, -1]

        return bbs, ccs, nimgs

    def get_embeding_vector(self, frame, image_format="BGR"):

        if isinstance(frame, str):
            frame = imread(frame)
            image_format = "RGB"

        bbs, ccs, nimgs = self.do_detect_in_image(frame, image_format)

        if len(bbs) == 0:
            return [], [], []

        emb = self.face_model((nimgs - 127.5) * 0.0078125).numpy()
        emb = normalize(emb)

        return emb, bbs, ccs



def draw_polyboxes(frame, bbs, ccs):
    for bb, cc in zip(bbs, ccs):
        # Red color for unknown, green for Recognized
        color = (0, 0, 255)
        left, up, right, down = bb
        cv2.line(frame, (left, up), (right, up), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, up), (right, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, down), (left, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (left, down), (left, up), color, 3, cv2.LINE_AA)

        xx, yy = np.max([bb[0] - 10, 10]), np.max([bb[1] - 10, 10])
        cv2.putText(frame, "Label: {}", (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame

if __name__ == "__main__":
    import sys
    import argparse

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, required=True, help="Saved basic_model file path, NOT model")
    args = parser.parse_known_args(sys.argv[1:])[0]

    model = Model(args.model_file)
    frame_path = "/workspace/data/test.jpg"
    frame = cv2.imread(frame_path)
    emb_vector, bbs, ccs = model.get_embeding_vector(frame=frame)
    print(emb_vector)
    draw_polyboxes(frame, bbs, ccs)
    cv2.imwrite("lol.png", frame)
