import cv2
import numpy as np
import logging
from skimage.io import imread
import sys
from pathlib import Path

# Add face-reidentification to path using absolute paths
from ..paths import get_face_reid_path
face_reid_path = get_face_reid_path()
print(f"Face ReID path: {face_reid_path}")
sys.path.insert(0, str(face_reid_path))

from models import SCRFD, ArcFace

logger = logging.getLogger(__name__)


class FaceModel:

    def __init__(
        self,
        det_weight: str = "det_10g.onnx",
        rec_weight: str = "w600k_r50.onnx",
        confidence_thresh: float = 0.5,
        input_size: tuple = (640, 640),
    ):
        from ..paths import get_weight_path
        
        det_path = get_weight_path(det_weight)
        rec_path = get_weight_path(rec_weight)
        
        try:
            self.det = SCRFD(str(det_path), input_size=input_size, conf_thres=confidence_thresh)
            self.face_model = ArcFace(str(rec_path))
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            raise RuntimeError(f"Face model initialization failed: {e}") from e

    def detect(self, image: np.ndarray, image_format: str = "BGR"):
        """Detect faces và trả về bboxes, kpss, confidence.

        Returns:
            bbs:  (N, 4) int array các bounding box
            ccs:  (N,)   float array confidence scores
            kpss: (N, 5, 2) keypoints — dùng để get_embedding
        """
        if image_format == "RGB":
            image = image[:, :, ::-1]  # về BGR cho SCRFD

        bboxes, kpss = self.det.detect(image, max_num=0)

        if len(bboxes) == 0:
            return [], [], []

        bbs = bboxes[:, :4].astype(np.int32)
        ccs = bboxes[:, 4].astype(np.float32)  # confidence score

        return bbs, ccs, kpss

    def get_embeding_vector(self, frame: np.ndarray, image_format: str = "BGR"):
        """Detect + extract embedding cho tất cả faces trong frame.

        Args:
            frame:        numpy array ảnh hoặc đường dẫn file
            image_format: "BGR" (mặc định OpenCV) hoặc "RGB"

        Returns:
            emb:  (N, embedding_size) normalized embeddings
            bbs:  (N, 4) bounding boxes
            ccs:  (N,)   confidence scores
        """
        if isinstance(frame, str):
            frame = imread(frame)
            image_format = "RGB"

        bbs, ccs, kpss = self.detect(frame, image_format)

        if len(bbs) == 0:
            return [], [], []

        if image_format == "RGB":
            frame = frame[:, :, ::-1]  # ArcFace cần BGR

        embeddings = []
        for kps in kpss:
            try:
                emb = self.face_model.get_embedding(frame, kps)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Error extracting embedding: {e}")
                continue

        if not embeddings:
            return [], [], []

        emb = np.stack(embeddings, axis=0)  # (N, embedding_size)

        return emb, bbs, ccs


def draw_polyboxes(
    frame: np.ndarray,
    bbs: list,
    ccs: list,
    names: list = None
) -> np.ndarray:
    """Vẽ bounding box lên frame.

    Args:
        frame: BGR image
        bbs:   list bounding boxes (left, up, right, down)
        ccs:   list confidence scores
        names: list tên tương ứng (optional)
    """
    for i, (bb, cc) in enumerate(zip(bbs, ccs)):
        color = (0, 255, 0) if names and names[i] != "Unknown" else (0, 0, 255)
        left, up, right, down = bb

        cv2.line(frame, (left, up),    (right, up),   color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, up),   (right, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, down), (left, down),  color, 3, cv2.LINE_AA)
        cv2.line(frame, (left, down),  (left, up),    color, 3, cv2.LINE_AA)

        label = names[i] if names else "Unknown"
        score = f"{cc:.2f}"
        xx = max(bb[0] - 10, 10)
        yy = max(bb[1] - 10, 10)
        cv2.putText(frame, f"{label} ({score})", (xx, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyModel:

    def __init__(self):

        # =========================
        # UNIVERSE
        # =========================

        self.score = ctrl.Antecedent(
            np.arange(0, 301, 1),
            'score'
        )

        self.confidence = ctrl.Antecedent(
            np.arange(0, 1.01, 0.01),
            'confidence'
        )

        self.decision = ctrl.Consequent(
            np.arange(0, 1.01, 0.01),
            'decision'
        )

        # =========================
        # MEMBERSHIP FUNCTIONS
        # =========================

        # Fingerprint Match Score

        self.score['low'] = fuzz.trapmf(
            self.score.universe,
            [0, 0, 80, 140]
        )

        self.score['medium'] = fuzz.trimf(
            self.score.universe,
            [100, 170, 240]
        )

        self.score['high'] = fuzz.trapmf(
            self.score.universe,
            [200, 250, 300, 300]
        )

        # Face Confidence

        self.confidence['low'] = fuzz.trapmf(
            self.confidence.universe,
            [0, 0, 0.25, 0.45]
        )

        self.confidence['medium'] = fuzz.trimf(
            self.confidence.universe,
            [0.35, 0.55, 0.75]
        )

        self.confidence['high'] = fuzz.trapmf(
            self.confidence.universe,
            [0.65, 0.8, 1, 1]
        )

        # Final Decision

        self.decision['reject'] = fuzz.trapmf(
            self.decision.universe,
            [0, 0, 0.2, 0.4]
        )

        self.decision['uncertain'] = fuzz.trimf(
            self.decision.universe,
            [0.3, 0.5, 0.7]
        )

        self.decision['accept'] = fuzz.trapmf(
            self.decision.universe,
            [0.6, 0.8, 1, 1]
        )

        # =========================
        # RULES
        # =========================

        self.rules = [

            ctrl.Rule(
                self.score['high'] &
                self.confidence['high'],
                self.decision['accept']
            ),

            ctrl.Rule(
                self.score['high'] &
                self.confidence['medium'],
                self.decision['accept']
            ),

            ctrl.Rule(
                self.score['medium'] &
                self.confidence['high'],
                self.decision['accept']
            ),

            ctrl.Rule(
                self.score['medium'] &
                self.confidence['medium'],
                self.decision['uncertain']
            ),

            ctrl.Rule(
                self.score['low'] &
                self.confidence['high'],
                self.decision['uncertain']
            ),

            ctrl.Rule(
                self.score['low'] &
                self.confidence['low'],
                self.decision['reject']
            ),

            ctrl.Rule(
                self.score['medium'] &
                self.confidence['low'],
                self.decision['reject']
            ),

            ctrl.Rule(
                self.score['high'] &
                self.confidence['low'],
                self.decision['uncertain']
            ),
        ]

        fuzzy_ctrl = ctrl.ControlSystem(self.rules)

        self.fuzzy_sim = ctrl.ControlSystemSimulation(
            fuzzy_ctrl,
            cache=False
        )

        print("[INFO] FUZZY model INIT SUCCESS")

    def make_decision(self, score, confidence):

        self.fuzzy_sim.reset()

        self.fuzzy_sim.input['score'] = float(score)

        self.fuzzy_sim.input['confidence'] = float(confidence)

        self.fuzzy_sim.compute()

        decision = self.fuzzy_sim.output['decision']

        print({
            "score": score,
            "confidence": confidence,
            "decision": decision
        })

        return decision


if __name__ == "__main__":

    model = FuzzyModel()

    # realistic test
    model.make_decision(240, 0.92)

    model.make_decision(120, 0.55)

    model.make_decision(40, 0.2)
    # model = FaceModel(
    #     det_weight="weights/det_10g.onnx",
    #     rec_weight="weights/w600k_r50.onnx",
    # )

    # frame = cv2.imread("test/lol.png")

    # start = time.time()
    # emb, bbs, ccs = model.get_embeding_vector(frame)
    # elapsed = time.time() - start

    # print(f"Detected {len(bbs)} face(s) | Time: {elapsed:.3f}s")
    # if len(emb) > 0:
    #     print(f"Embedding shape: {emb.shape}")

    # frame = draw_polyboxes(frame, bbs, ccs)
    # cv2.imwrite("output.jpg", frame)
  