import json
import logging
import urllib.request

import numpy as np

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
        self.base_url = f"http://{host}:{port}"
        self._ensure_collection()

    # -- Internal -------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            content = response.read().decode("utf-8")
            return json.loads(content) if content else {}

    def _filter(self, key: str, value) -> dict:
        return {
            "must": [
                {
                    "key": key,
                    "match": {"value": value},
                }
            ]
        }

    def _ensure_collection(self) -> None:
        collections = self._request("GET", "/collections")
        existing = [
            item["name"]
            for item in collections.get("result", {}).get("collections", [])
        ]

        if self.collection_name not in existing:
            self._request(
                "PUT",
                f"/collections/{self.collection_name}",
                {
                    "vectors": {
                        "size": self.embedding_size,
                        "distance": "Cosine",
                    }
                },
            )
            logger.info("Created collection: '%s'", self.collection_name)
            return

        info = self._request("GET", f"/collections/{self.collection_name}")
        count = info.get("result", {}).get("points_count") or 0
        logger.info(
            "Connected to collection: '%s' | points: %s",
            self.collection_name,
            count,
        )

    def _next_id(self) -> int:
        return self.count()

    # -- Write ----------------------------------------------------------------

    def add(
        self,
        embedding: np.ndarray,
        name: str,
        fingerprint_position: int = -1,
    ) -> None:
        self._request(
            "PUT",
            f"/collections/{self.collection_name}/points?wait=true",
            {
                "points": [
                    {
                        "id": self._next_id(),
                        "vector": embedding.tolist(),
                        "payload": {
                            "name": name,
                            "fingerprint_position": fingerprint_position,
                        },
                    }
                ]
            },
        )
        logger.info(
            "Added: '%s' | fingerprint_position=%s",
            name,
            fingerprint_position,
        )

    def add_batch(
        self,
        embeddings: list[np.ndarray],
        names: list[str],
        fingerprint_positions: list[int] | None = None,
    ) -> None:
        if fingerprint_positions is None:
            fingerprint_positions = [-1] * len(names)

        offset = self._next_id()
        points = [
            {
                "id": offset + i,
                "vector": emb.tolist(),
                "payload": {
                    "name": name,
                    "fingerprint_position": fp_pos,
                },
            }
            for i, (emb, name, fp_pos) in enumerate(
                zip(embeddings, names, fingerprint_positions)
            )
        ]
        self._request(
            "PUT",
            f"/collections/{self.collection_name}/points?wait=true",
            {"points": points},
        )
        logger.info("Added batch: %s faces", len(points))

    def delete(self, name: str) -> None:
        self._request(
            "POST",
            f"/collections/{self.collection_name}/points/delete?wait=true",
            {"filter": self._filter("name", name)},
        )
        logger.info("Deleted: '%s'", name)

    # -- Read -----------------------------------------------------------------

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.4,
    ) -> list[tuple[str, float, int]]:
        if self.is_empty():
            logger.warning("Collection rỗng, không thể search.")
            return [("Unknown", 0.0, -1)]

        response = self._request(
            "POST",
            f"/collections/{self.collection_name}/points/search",
            {
                "vector": embedding.tolist(),
                "limit": top_k,
                "with_payload": True,
            },
        )
        hits = response.get("result", [])

        if not hits:
            return [("Unknown", 0.0, -1)]

        filtered = [
            (
                hit.get("payload", {}).get("name", "Unknown"),
                float(hit.get("score", 0.0)),
                int(hit.get("payload", {}).get("fingerprint_position", -1)),
            )
            for hit in hits
            if hit.get("score", 0.0) >= threshold
        ]

        if filtered:
            return filtered

        return [("Unknown", float(hits[0].get("score", 0.0)), -1)]

    # -- Utils ----------------------------------------------------------------

    def _scroll(
        self,
        scroll_filter: dict | None = None,
        limit: int = 1,
    ) -> list[dict]:
        body = {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if scroll_filter is not None:
            body["filter"] = scroll_filter

        response = self._request(
            "POST",
            f"/collections/{self.collection_name}/points/scroll",
            body,
        )
        return response.get("result", {}).get("points", [])

    def get_name_by_fp_position(self, fp_position: int) -> str:
        if fp_position < 0:
            return "Unknown"

        results = self._scroll(
            scroll_filter=self._filter("fingerprint_position", fp_position),
            limit=1,
        )
        if not results:
            return "Unknown"
        return results[0].get("payload", {}).get("name", "Unknown")

    def get_fingerprint_position(self, name: str) -> int:
        results = self._scroll(
            scroll_filter=self._filter("name", name),
            limit=1,
        )
        if not results:
            return -1
        return int(results[0].get("payload", {}).get("fingerprint_position", -1))

    def count(self) -> int:
        response = self._request(
            "POST",
            f"/collections/{self.collection_name}/points/count",
            {"exact": True},
        )
        return int(response.get("result", {}).get("count", 0))

    def list_names(self) -> list[str]:
        results = self._scroll(limit=10_000)
        names = {
            item.get("payload", {}).get("name")
            for item in results
            if item.get("payload")
        }
        return sorted(name for name in names if name)

    def list_all(self) -> list[dict]:
        results = self._scroll(limit=10_000)
        return [
            {
                "id": item.get("id"),
                "name": item.get("payload", {}).get("name", ""),
                "fingerprint_position": item.get("payload", {}).get(
                    "fingerprint_position",
                    -1,
                ),
            }
            for item in results
            if item.get("payload")
        ]

    def clear(self) -> None:
        self._request("DELETE", f"/collections/{self.collection_name}")
        self._ensure_collection()
        logger.info("Cleared collection: '%s'", self.collection_name)

    def is_empty(self) -> bool:
        return self.count() == 0

    def __repr__(self) -> str:
        return (
            f"Qdrant_db(collection='{self.collection_name}', "
            f"size={self.embedding_size}, count={self.count()})"
        )
