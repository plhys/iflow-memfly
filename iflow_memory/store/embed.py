"""Embedding client — 深度回忆的向量生成层。

支持三种后端：
- onnx: 本地 ONNX 模型（fastembed），零外部依赖，离线可用
- api:  调用 OpenAI 兼容的 embedding API（用户可自选模型）
- off:  不生成向量，纯 FTS5 搜索（现有行为）

用法：
    embedder = Embedder(config)
    await embedder.init()
    vec = await embedder.embed("一段文本")        # -> list[float] | None
    vecs = await embedder.embed_batch(["a", "b"]) # -> list[list[float]] | None
"""

import logging
from typing import Optional

import httpx

from ..config import MemoryConfig

logger = logging.getLogger("iflow-memory")


class Embedder:
    """统一的 embedding 客户端，根据配置选择后端。"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.backend = config.embed_backend  # "onnx" | "api" | "off"
        self._onnx_model = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._dim: int = 0  # embedding 维度，init 后确定
        self._available = False

    @property
    def available(self) -> bool:
        """是否可用（初始化成功且非 off 模式）。"""
        return self._available

    @property
    def dimension(self) -> int:
        """当前 embedding 维度。"""
        return self._dim

    async def init(self) -> bool:
        """初始化后端。返回 True 表示可用。

        失败时不抛异常，只记日志并标记为不可用（降级到 FTS5）。
        """
        if self.backend == "off":
            logger.info("[深度回忆] embedding 已关闭，使用纯 FTS5 搜索")
            return False

        if self.backend == "onnx":
            return await self._init_onnx()
        elif self.backend == "api":
            return await self._init_api()
        else:
            logger.warning(f"[深度回忆] 未知 embedding 后端: {self.backend}，降级为 off")
            self.backend = "off"
            return False

    async def _init_onnx(self) -> bool:
        """初始化本地 ONNX embedding（fastembed）。"""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            logger.warning(
                "[深度回忆] fastembed 未安装，无法使用 onnx 后端。"
                "安装: pip install fastembed。降级为 FTS5 搜索"
            )
            self.backend = "off"
            return False

        try:
            model_name = self.config.embed_model or "BAAI/bge-small-en-v1.5"
            self._onnx_model = TextEmbedding(model_name=model_name)
            # 探测维度
            test = list(self._onnx_model.embed(["test"]))[0]
            self._dim = len(test)
            self._available = True
            logger.info(f"[深度回忆] ONNX 后端就绪: {model_name}, 维度={self._dim}")
            return True
        except Exception as e:
            logger.warning(f"[深度回忆] ONNX 初始化失败: {e}，降级为 FTS5 搜索")
            self.backend = "off"
            return False

    async def _init_api(self) -> bool:
        """初始化 API embedding 客户端。"""
        if not self.config.embed_api_url:
            logger.warning("[深度回忆] embed_api_url 未配置，降级为 FTS5 搜索")
            self.backend = "off"
            return False

        self._http_client = httpx.AsyncClient(timeout=30.0)

        # 探测维度：发一个测试请求
        try:
            test_vec = await self._call_api(["dimension probe"])
            if test_vec and len(test_vec) > 0:
                self._dim = len(test_vec[0])
                self._available = True
                logger.info(
                    f"[深度回忆] API 后端就绪: {self.config.embed_api_url}, "
                    f"model={self.config.embed_model}, 维度={self._dim}"
                )
                return True
            else:
                raise ValueError("API 返回空向量")
        except Exception as e:
            logger.warning(f"[深度回忆] API 初始化失败: {e}，降级为 FTS5 搜索")
            self.backend = "off"
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            return False

    async def embed(self, text: str) -> Optional[list[float]]:
        """生成单条文本的 embedding。失败返回 None。"""
        if not self._available:
            return None
        result = await self.embed_batch([text])
        return result[0] if result else None

    async def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        """批量生成 embedding。失败返回 None。"""
        if not self._available or not texts:
            return None

        try:
            if self.backend == "onnx":
                return self._embed_onnx(texts)
            elif self.backend == "api":
                return await self._call_api(texts)
        except Exception as e:
            logger.error(f"[深度回忆] embedding 生成失败: {e}")
            return None

    def _embed_onnx(self, texts: list[str]) -> list[list[float]]:
        """ONNX 本地推理。"""
        embeddings = list(self._onnx_model.embed(texts))
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI 兼容的 embedding API。"""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")

        base_url = self.config.embed_api_url.rstrip("/")
        if not base_url.endswith("/embeddings"):
            if base_url.endswith("/v1"):
                url = f"{base_url}/embeddings"
            else:
                url = f"{base_url}/v1/embeddings"
        else:
            url = base_url

        headers = {"Content-Type": "application/json"}
        if self.config.embed_api_key:
            headers["Authorization"] = f"Bearer {self.config.embed_api_key}"

        resp = await self._http_client.post(
            url,
            headers=headers,
            json={
                "model": self.config.embed_model or "text-embedding-ada-002",
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI 格式: {"data": [{"embedding": [...], "index": 0}, ...]}
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def close(self) -> None:
        """释放资源。"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._onnx_model = None
        self._available = False
