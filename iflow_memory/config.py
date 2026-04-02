"""iFlow MemFly configuration management."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iflow-memory")

DEFAULT_CONFIG_DIR = Path.home() / ".iflow-memory"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


@dataclass
class ModelPreset:
    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class MemoryConfig:
    # 监听路径
    acp_sessions_dir: str = str(Path.home() / ".iflow" / "acp" / "sessions")
    cli_sessions_dir: str = str(Path.home() / ".iflow" / "projects")
    memory_dir: str = str(Path.home() / ".iflow-memory" / "data")

    # 摘要策略: "on_compress" | "interval" | "idle"
    strategy: str = "interval"
    interval_seconds: int = 300  # 普通模式: 5分钟

    # 索引
    index_recent_lines: int = 50

    # 功能开关（需要 LLM 的功能，可按需关闭以节省 token）
    # - index_line: L1 索引短句（轻量，推荐开启）
    # - summary: L2 结构化摘要
    # - classify: 分类记忆提取（推荐开启）
    # - atmosphere: 对话氛围快照
    # - daily_recap: 每日工作回顾（最耗 token）
    # - vector_search: 深度回忆 — 向量搜索（需要 embedding 后端）
    features: dict = field(default_factory=lambda: {
        "index_line": True,
        "summary": True,
        "classify": True,
        "atmosphere": True,
        "state_snapshot": True,
        "daily_recap": True,
        "vector_search": True,
        "knowledge_graph": True,
        "daily_briefing": True,
        "llm_dream": False,  # LLM 深度记忆整合（子代理做梦），默认关闭
    })

    # 深度回忆 — embedding 配置
    # embed_backend: "onnx" (本地 fastembed) | "api" (OpenAI 兼容) | "off" (纯 FTS5)
    embed_backend: str = "off"
    embed_api_url: str = ""     # API 模式的 endpoint（如 https://api.openai.com）
    embed_api_key: str = ""     # API 模式的密钥
    embed_model: str = ""       # 模型名（onnx 默认 bge-small-en-v1.5，api 默认 text-embedding-ada-002）
    embed_dim: int = 0          # embedding 维度（0 = 不启用向量搜索）

    # Web 服务端口
    web_port: int = 18765

    # 注入目标
    agents_md_paths: list = field(default_factory=lambda: [
        str(Path.home() / ".iflow" / "AGENTS.md"),
    ])

    # 模型配置
    model_mode: str = "follow"  # "follow" | "custom"
    active_preset: str = "default"
    model_presets: dict = field(default_factory=lambda: {
        "default": {
            "name": "default",
            "base_url": "",
            "api_key": "",
            "model": "glm-5",
        }
    })

    # 宿主智能体配置路径（follow 模式下从宿主读取模型信息）
    iflow_bot_config: str = str(Path.home() / ".iflow-bot" / "config.json")

    def get_active_model(self) -> ModelPreset:
        """获取当前活跃的模型配置。"""
        if self.model_mode == "follow":
            return self._read_bot_model()
        preset_data = self.model_presets.get(self.active_preset, {})
        return ModelPreset(**preset_data)

    def _read_bot_model(self) -> ModelPreset:
        """从 iflow-bot config 和 iflow settings 读取当前模型。"""
        try:
            # 从 iflow-bot config 读模型名
            model = "glm-5"
            try:
                with open(self.iflow_bot_config) as f:
                    cfg = json.load(f)
                model = cfg.get("driver", {}).get("model", model)
            except (OSError, json.JSONDecodeError):
                pass

            # 从 iflow settings 读 API 信息
            base_url, api_key = "", ""
            iflow_settings = Path.home() / ".iflow" / "settings.json"
            if iflow_settings.exists():
                with open(iflow_settings) as f:
                    settings = json.load(f)
                base_url = settings.get("baseUrl", "")
                api_key = settings.get("apiKey", "")
                # iFlow CLI 的 settings.json 有时 apiKey 字段存的是 URL
                if api_key.startswith("http"):
                    api_key = ""

            return ModelPreset(
                name="follow",
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        except Exception as e:
            logger.warning(f"Failed to read bot model config: {e}")
            preset_data = self.model_presets.get("default", {})
            return ModelPreset(**preset_data)


def load_config(path: Optional[Path] = None) -> MemoryConfig:
    """加载配置文件。"""
    config_file = path or DEFAULT_CONFIG_FILE
    if config_file.exists():
        try:
            with open(config_file) as f:
                data = json.load(f)
            return MemoryConfig(**{
                k: v for k, v in data.items()
                if k in MemoryConfig.__dataclass_fields__
            })
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    return MemoryConfig()


def save_config(config: MemoryConfig, path: Optional[Path] = None) -> None:
    """保存配置文件。"""
    config_file = path or DEFAULT_CONFIG_FILE
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
