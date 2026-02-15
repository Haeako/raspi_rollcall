#!/usr/bin/env python3

import cv2
import sys

import numpy as np
from skimage.io import imread
from insightface.app import FaceAnalysis

class Model:

    def __init__(self, model_file=None, ctx_id=0, det_size=(640, 640)):
        # model_file is kept for backward compatibility with old GhostFaceNet-based constructor.
        self.app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def do_detect_in_image(self, image, image_format="BGR"):
        imm_bgr = image if image_format == "BGR" else image[:, :, ::-1]
        faces = self.app.get(imm_bgr)

        if len(faces) == 0:
            return np.array([]), np.array([]), []

        bbs = np.array([face.bbox.astype("int") for face in faces])
        ccs = np.array([face.det_score for face in faces])

        return bbs, ccs, faces

    def get_embeding_vector(self, frame, image_format="BGR"):

        if isinstance(frame, str):
            frame = imread(frame)
            image_format = "RGB"

        bbs, ccs, faces = self.do_detect_in_image(frame, image_format)

        if len(bbs) == 0:
            return [], [], []

        emb = np.array([face.normed_embedding for face in faces])

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
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Unused with InsightFace. Kept for compatibility.")
    args = parser.parse_known_args(sys.argv[1:])[0]

    model = Model(args.model_file)
    frame_path = "/workspace/raspi_rollcall/data/test.jpg"
    frame = cv2.imread(frame_path)
    emb_vector, bbs, ccs = model.get_embeding_vector(frame=frame)
    print(emb_vector)
    draw_polyboxes(frame, bbs, ccs)
    cv2.imwrite("lol.png", frame)
