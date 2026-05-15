import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

logger = logging.getLogger(__name__)


class Qdrant_db:

    def __init__(
        self,
        host: str = "qdrant",
        port: int = 6333,
        collection_name: str = "faces",
        embedding_size: int = 512,
    ):
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: '{self.collection_name}'")
        else:
            info  = self.client.get_collection(self.collection_name)
            count = info.points_count or 0
            logger.info(
                f"Connected to collection: '{self.collection_name}' | points: {count}"
            )

    def _next_id(self) -> int:
        return self.client.count(self.collection_name).count

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        embedding: np.ndarray,
        name: str,
        fingerprint_position: int = -1,   # idx AS608, -1 = chưa enroll vân tay
    ) -> None:
        """
        Thêm 1 embedding vào collection.

        Args:
            embedding:            vector khuôn mặt (512-d).
            name:                 tên người.
            fingerprint_position: vị trí template trong AS608 (-1 nếu chưa có).
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=self._next_id(),
                vector=embedding.tolist(),
                payload={
                    "name":                 name,
                    "fingerprint_position": fingerprint_position,
                },
            )],
        )
        logger.info(
            f"Added: '{name}' | fingerprint_position={fingerprint_position}"
        )

    def add_batch(
        self,
        embeddings: list[np.ndarray],
        names: list[str],
        fingerprint_positions: list[int] | None = None,
    ) -> None:
        """
        Thêm nhiều embedding cùng lúc.

        Args:
            fingerprint_positions: list idx AS608 tương ứng; None → toàn bộ là -1.
        """
        if fingerprint_positions is None:
            fingerprint_positions = [-1] * len(names)

        offset = self._next_id()
        points = [
            PointStruct(
                id=offset + i,
                vector=emb.tolist(),
                payload={
                    "name":                 name,
                    "fingerprint_position": fp_pos,
                },
            )
            for i, (emb, name, fp_pos) in enumerate(
                zip(embeddings, names, fingerprint_positions)
            )
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added batch: {len(points)} faces")

    def delete(self, name: str) -> None:
        """Xóa tất cả vector theo tên."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
        )
        logger.info(f"Deleted: '{name}'")

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.4,
    ) -> list[tuple[str, float, int]]:
        """
        Tìm kiếm khuôn mặt gần nhất.

        Returns:
            List các tuple (name, score, fingerprint_position).
            Trả về [("Unknown", 0.0, -1)] khi không tìm thấy.
        """
        if self.is_empty():
            logger.warning("Collection rỗng, không thể search.")
            return [("Unknown", 0.0, -1)]

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

        if not hits:
            return [("Unknown", 0.0, -1)]

        filtered = [
            (
                h.payload.get("name", "Unknown"),
                float(h.score),
                int(h.payload.get("fingerprint_position", -1)),
            )
            for h in hits
            if h.score >= threshold
        ]

        return filtered if filtered else [("Unknown", 0.0, -1)]

    def search_or_enroll(
        self,
        embedding: np.ndarray,
        threshold: float = 0.4,
        fingerprint_position: int = -1,
    ) -> tuple[str, float, int]:
        """
        Tìm khuôn mặt. Nếu không có trong DB → hỏi user qua terminal
        và enroll nếu được đồng ý.

        Args:
            embedding:            vector khuôn mặt cần tìm.
            threshold:            ngưỡng cosine similarity.
            fingerprint_position: idx vân tay AS608 đã enroll (-1 nếu chưa có).

        Returns:
            Tuple (name, score, fingerprint_position) của kết quả cuối cùng.
        """
        results = self.search(embedding, top_k=1, threshold=threshold)
        name, score, fp_pos = results[0]

        if name != "Unknown":
            logger.info(f"Nhận diện: '{name}' (score={score:.3f}, fp_pos={fp_pos})")
            return name, score, fp_pos

        # ── Không tìm thấy → hỏi enroll ──────────────────────────────────
        print(f"\n[DB] Khuôn mặt lạ (best score={score:.3f} < threshold={threshold})")
        ans = input("Enroll khuôn mặt này không? (y/n): ").strip().lower()

        if ans != "y":
            print("[DB] Bỏ qua enroll.")
            return "Unknown", 0.0, -1

        new_name = input("Nhập tên: ").strip()
        if not new_name:
            print("[DB] Tên rỗng, bỏ qua.")
            return "Unknown", 0.0, -1

        # Hỏi fingerprint position nếu chưa truyền vào
        if fingerprint_position == -1:
            fp_input = input(
                "Nhập vị trí vân tay trong AS608 (-1 nếu chưa enroll): "
            ).strip()
            try:
                fingerprint_position = int(fp_input)
            except ValueError:
                fingerprint_position = -1

        self.add(embedding, new_name, fingerprint_position)
        print(
            f"[DB] Đã lưu '{new_name}' "
            f"(fingerprint_position={fingerprint_position})"
        )
        return new_name, 1.0, fingerprint_position

    # ── Utils ─────────────────────────────────────────────────────────────────

    def get_name_by_fp_position(self, fp_position: int) -> str:
        """
        Tra tên người theo fingerprint_position (slot AS608).
        Trả về "Unknown" nếu không tìm thấy.
        """
        if fp_position < 0:
            return "Unknown"

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="fingerprint_position",
                    match=MatchValue(value=fp_position),
                )]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return "Unknown"
        return results[0].payload.get("name", "Unknown")

    def get_fingerprint_position(self, name: str) -> int:
        """
        Lấy fingerprint_position đã lưu của một người theo tên.
        Trả về -1 nếu không tìm thấy.
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return -1
        return int(results[0].payload.get("fingerprint_position", -1))

    def count(self) -> int:
        return self.client.count(self.collection_name).count

    def list_names(self) -> list[str]:
        """Danh sách tên unique trong collection."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        names = list({r.payload.get("name") for r in results if r.payload})
        return sorted(names)

    def list_all(self) -> list[dict]:
        """
        Trả về toàn bộ records dạng list dict:
        [{"id": int, "name": str, "fingerprint_position": int}, ...]
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id":                   r.id,
                "name":                 r.payload.get("name", ""),
                "fingerprint_position": r.payload.get("fingerprint_position", -1),
            }
            for r in results
            if r.payload
        ]

    def clear(self) -> None:
        """Xóa toàn bộ data, giữ lại collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.info(f"Cleared collection: '{self.collection_name}'")

    def is_empty(self) -> bool:
        return self.count() == 0

    def __repr__(self) -> str:
        return (
            f"Qdrant_db(collection='{self.collection_name}', "
            f"size={self.embedding_size}, count={self.count()})"
        )