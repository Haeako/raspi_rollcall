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
    SCORE_MAX = 300.0
    ACCEPT_THRESHOLD = 0.65
    RETRY_THRESHOLD = 0.3

    def __init__(self) -> None:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        self.score["low"] = fuzz.trapmf(self.score.universe, [0, 0, 90, 170])
        self.score["medium"] = fuzz.trimf(self.score.universe, [100, 170, 240])
        self.score["high"] = fuzz.trapmf(self.score.universe, [170, 230, 300, 300])
        
        self.confidence["low"] = fuzz.trapmf(self.confidence.universe, [0, 0, 0.30, 0.55])
        self.confidence["medium"] = fuzz.trimf(self.confidence.universe, [0.35, 0.60, 0.85])
        self.confidence["high"] = fuzz.trapmf(self.confidence.universe, [0.60, 0.75, 1.0, 1.0])
        
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
            [0, 0, 0.2, 0.35],
        )
        self.decision["uncertain"] = fuzz.trimf(
            self.decision.universe,
            [0.25, 0.5, 0.75],
        )
        self.decision["accept"] = fuzz.trapmf(
            self.decision.universe,
            [0.65, 0.8, 1, 1],
        )

        rules = [
            ctrl.Rule(
                self.score["low"] & self.confidence["low"],
                self.decision["reject"],
            ),
            ctrl.Rule(
                self.score["low"] & self.confidence["medium"],
                self.decision["reject"],
            ),
            ctrl.Rule(
                self.score["low"] & self.confidence["high"],
                self.decision["uncertain"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["low"],
                self.decision["reject"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["medium"],
                self.decision["uncertain"],
            ),
            ctrl.Rule(
                self.score["medium"] & self.confidence["high"],
                self.decision["accept"],
            ),
            ctrl.Rule(
                self.score["high"] & self.confidence["low"],
                self.decision["uncertain"],
            ),
            ctrl.Rule(
                self.score["high"] & self.confidence["medium"],
                self.decision["accept"],
            ),
            ctrl.Rule(
                self.score["high"] & self.confidence["high"],
                self.decision["accept"],
            ),
        ]

        fuzzy_ctrl = ctrl.ControlSystem(rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl, cache=False)
        logger.info("Fuzzy model initialized")

    def make_decision(self, score, confidence) -> float:
        score_value = max(0.0, min(float(score), self.SCORE_MAX))
        confidence_value = max(0.0, min(float(confidence), 1.0))

        self.fuzzy_sim.reset()
        self.fuzzy_sim.input["score"] = score_value
        self.fuzzy_sim.input["confidence"] = confidence_value
        self.fuzzy_sim.compute()
        return float(self.fuzzy_sim.output["decision"])

    def classify_decision(self, decision_score: float) -> str:
        if decision_score >= self.ACCEPT_THRESHOLD:
            return "Accept"
        if decision_score < self.RETRY_THRESHOLD:
            return "Reject"
        return "Uncertain"


def main() -> None:
    model = FuzzyModel()

    test_cases = [
        # score low
        ("Low", "Low", 40, 0.20),
        ("Low", "Medium", 40, 0.55),
        ("Low", "High", 40, 0.90),

        # score medium
        ("Medium", "Low", 170, 0.20),
        ("Medium", "Medium", 170, 0.55),
        ("Medium", "High", 170, 0.90),

        # score high
        ("High", "Low", 260, 0.20),
        ("High", "Medium", 260, 0.55),
        ("High", "High", 260, 0.90),
    ]

    print(f"{'Case':<5} {'Score':<8} {'Conf':<8} {'ScoreVal':<9} {'ConfVal':<8} {'Decision':<10} {'Label'}")
    print("-" * 70)

    for i, (score_label, conf_label, score, confidence) in enumerate(test_cases, start=1):
        decision = model.make_decision(score, confidence)

        label = model.classify_decision(decision)

        print(
            f"R{i:<4} "
            f"{score_label:<8} "
            f"{conf_label:<8} "
            f"{score:<9} "
            f"{confidence:<8.2f} "
            f"{decision:<10.4f} "
            f"{label}"
        )

if __name__ == "__main__":
    main()
