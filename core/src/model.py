"""Face recognition and decision models."""

import logging
import sys

import cv2
import numpy as np

from ..paths import get_face_reid_path, get_weight_path

logger = logging.getLogger(__name__)

FACE_REID_PATH = get_face_reid_path()
if str(FACE_REID_PATH) not in sys.path:
    sys.path.insert(0, str(FACE_REID_PATH))


class FaceModel:
    def __init__(
        self,
        det_weight: str = "det_10g.onnx",
        rec_weight: str = "w600k_r50.onnx",
        confidence_thresh: float = 0.5,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        from models import ArcFace, SCRFD

        det_path = get_weight_path(det_weight)
        rec_path = get_weight_path(rec_weight)

        try:
            self.det = SCRFD(
                str(det_path),
                input_size=input_size,
                conf_thres=confidence_thresh,
            )
            self.face_model = ArcFace(str(rec_path))
        except Exception as error:
            logger.error("Failed to load face models: %s", error)
            raise RuntimeError(f"Face model initialization failed: {error}") from error

    def detect(self, image: np.ndarray, image_format: str = "BGR"):
        """Return face boxes, confidence scores, and keypoints."""
        if image_format == "RGB":
            image = image[:, :, ::-1]

        bboxes, keypoints = self.det.detect(image, max_num=0)
        if len(bboxes) == 0:
            return [], [], []

        boxes = bboxes[:, :4].astype(np.int32)
        confidences = bboxes[:, 4].astype(np.float32)
        return boxes, confidences, keypoints

    def get_embedding_vector(self, frame: np.ndarray | str, image_format: str = "BGR"):
        """Detect faces and return embeddings, boxes, and confidence scores."""
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            image_format = "BGR"
            if frame is None:
                raise ValueError(f"Could not read image: {frame}")

        boxes, confidences, keypoints = self.detect(frame, image_format)
        if len(boxes) == 0:
            return [], [], []

        if image_format == "RGB":
            frame = frame[:, :, ::-1]

        embeddings = []
        for points in keypoints:
            try:
                embeddings.append(self.face_model.get_embedding(frame, points))
            except Exception as error:
                logger.warning("Error extracting embedding: %s", error)

        if not embeddings:
            return [], [], []

        return np.stack(embeddings, axis=0), boxes, confidences

    def get_embeding_vector(self, frame: np.ndarray | str, image_format: str = "BGR"):
        """Backward-compatible alias for the original misspelled method name."""
        return self.get_embedding_vector(frame, image_format)


def draw_polyboxes(
    frame: np.ndarray,
    bbs: list,
    ccs: list,
    names: list | None = None,
) -> np.ndarray:
    """Draw face boxes and labels on a BGR frame."""
    for index, (box, confidence) in enumerate(zip(bbs, ccs)):
        label = names[index] if names else "Unknown."
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        left, top, right, bottom = [int(value) for value in box]

        cv2.line(frame, (left, top), (right, top), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, top), (right, bottom), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, bottom), (left, bottom), color, 3, cv2.LINE_AA)
        cv2.line(frame, (left, bottom), (left, top), color, 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (max(left - 10, 10), max(top - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

    return frame


class FuzzyModel:
    def __init__(self) -> None:
        try:
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
        except ImportError:
            self.fuzzy_sim = None
            logger.warning("skfuzzy not installed; using simple decision fallback")
            return

        self.score = ctrl.Antecedent(np.arange(0, 301, 1), "score")
        self.confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "confidence")
        self.decision = ctrl.Consequent(np.arange(0, 1.01, 0.01), "decision")

        self.score["low"] = fuzz.trapmf(self.score.universe, [0, 0, 80, 140])
        self.score["medium"] = fuzz.trimf(self.score.universe, [100, 170, 240])
        self.score["high"] = fuzz.trapmf(self.score.universe, [200, 250, 300, 300])

        self.confidence["low"] = fuzz.trapmf(
            self.confidence.universe,
            [0, 0, 0.25, 0.45],
        )
        self.confidence["medium"] = fuzz.trimf(
            self.confidence.universe,
            [0.35, 0.55, 0.75],
        )
        self.confidence["high"] = fuzz.trapmf(
            self.confidence.universe,
            [0.65, 0.8, 1, 1],
        )

        self.decision["reject"] = fuzz.trapmf(
            self.decision.universe,
            [0, 0, 0.2, 0.4],
        )
        self.decision["uncertain"] = fuzz.trimf(
            self.decision.universe,
            [0.3, 0.5, 0.7],
        )
        self.decision["accept"] = fuzz.trapmf(self.decision.universe, [0.6, 0.8, 1, 1])

        rules = [
            ctrl.Rule(
                self.score["low"] & self.confidence["medium"],
                self.decision["reject"],
            )
            ctrl.Rule(
                self.score["high"] & self.confidence["high"],
                self.decision["accept"],
            ),
            ctrl.Rule(
                self.score["high"] & self.confidence["medium"],
                self.decision["accept"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["high"],
                self.decision["accept"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["medium"],
                self.decision["uncertain"],
            ),
            ctrl.Rule(
                self.score["low"] & self.confidence["high"],
                self.decision["uncertain"],
            ),
            ctrl.Rule(
                self.score["low"] & self.confidence["low"],
                self.decision["reject"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["low"],
                self.decision["reject"],
            ),
            ctrl.Rule(
                self.score["high"] & self.confidence["low"],
                self.decision["uncertain"],
            ),
        ]

        fuzzy_ctrl = ctrl.ControlSystem(rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl, cache=False)
        logger.info("Fuzzy model initialized")

    def make_decision(self, score, confidence) -> float:
        if self.fuzzy_sim is None:
            score_norm = max(0.0, min(float(score) / 300.0, 1.0))
            conf_norm = max(0.0, min(float(confidence), 1.0))
            return (score_norm * 0.55) + (conf_norm * 0.45)

        self.fuzzy_sim.reset()
        self.fuzzy_sim.input["score"] = float(score)
        self.fuzzy_sim.input["confidence"] = float(confidence)
        self.fuzzy_sim.compute()
        return float(self.fuzzy_sim.output["decision"])


def main() -> None:
    model = FuzzyModel()
    for score, confidence in [(240, 0.92), (120, 0.55), (40, 0.2)]:
        print(model.make_decision(score, confidence))


if __name__ == "__main__":
    main()
