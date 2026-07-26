from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import PRODUCTION_CONFIG_PATH


SUPPORTED_CONFIGURATIONS = {
    "faiss_only",
    "hybrid_no_ce",
    "cross_encoder_only",
    "hybrid_with_ce",
}


@dataclass(frozen=True)
class ProductionMode:
    """Cấu hình cho một chế độ recommendation."""

    name: str
    configuration: str
    description: str
    use_cross_encoder: bool
    use_hybrid_ranking: bool


@dataclass(frozen=True)
class RetrievalSettings:
    """Giới hạn retrieval và số kết quả trả về."""

    candidate_k: int
    default_top_k: int
    min_top_k: int
    max_top_k: int


@dataclass(frozen=True)
class ProductionSettings:
    """Toàn bộ cấu hình production đã được validate."""

    default_mode: str
    modes: dict[str, ProductionMode]
    retrieval: RetrievalSettings
    default_include_explanation: bool

    def get_mode(self, mode_name: str | None = None) -> ProductionMode:
        """Lấy mode được yêu cầu hoặc mode mặc định."""

        selected_name = mode_name or self.default_mode

        try:
            return self.modes[selected_name]
        except KeyError as error:
            allowed_modes = ", ".join(sorted(self.modes))

            raise ValueError(
                f"Mode không hợp lệ: {selected_name!r}. "
                f"Các mode được hỗ trợ: {allowed_modes}."
            ) from error

    def validate_top_k(self, top_k: int | None = None) -> int:
        """Kiểm tra và trả về top_k hợp lệ."""

        selected_top_k = (
            self.retrieval.default_top_k
            if top_k is None
            else top_k
        )

        if not (
            self.retrieval.min_top_k
            <= selected_top_k
            <= self.retrieval.max_top_k
        ):
            raise ValueError(
                "top_k phải nằm trong khoảng "
                f"{self.retrieval.min_top_k}–"
                f"{self.retrieval.max_top_k}. "
                f"Giá trị nhận được: {selected_top_k}."
            )

        return selected_top_k


def _read_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML và đảm bảo root là mapping."""

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy production config: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "production.yaml phải chứa một YAML mapping."
        )

    return raw_config


def _require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Lấy một mapping bắt buộc trong YAML."""

    value = config.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"production.yaml thiếu mapping hợp lệ: {key}"
        )

    return value


def load_production_settings(
    path: Path = PRODUCTION_CONFIG_PATH,
) -> ProductionSettings:
    """Đọc và validate production.yaml."""

    raw_config = _read_yaml(path)

    default_mode = raw_config.get("default_mode")

    if not isinstance(default_mode, str) or not default_mode.strip():
        raise ValueError(
            "default_mode phải là một chuỗi không rỗng."
        )

    raw_modes = _require_mapping(
        raw_config,
        "modes",
    )

    modes: dict[str, ProductionMode] = {}

    for mode_name, raw_mode in raw_modes.items():
        if not isinstance(raw_mode, dict):
            raise ValueError(
                f"Mode {mode_name!r} phải là một mapping."
            )

        configuration = raw_mode.get("configuration")

        if configuration not in SUPPORTED_CONFIGURATIONS:
            raise ValueError(
                f"Configuration không hợp lệ cho mode "
                f"{mode_name!r}: {configuration!r}."
            )

        modes[str(mode_name)] = ProductionMode(
            name=str(mode_name),
            configuration=str(configuration),
            description=str(
                raw_mode.get("description", "")
            ).strip(),
            use_cross_encoder=bool(
                raw_mode.get("use_cross_encoder", False)
            ),
            use_hybrid_ranking=bool(
                raw_mode.get("use_hybrid_ranking", False)
            ),
        )

    if default_mode not in modes:
        raise ValueError(
            "default_mode không tồn tại trong modes: "
            f"{default_mode!r}"
        )

    raw_retrieval = _require_mapping(
        raw_config,
        "retrieval",
    )

    retrieval = RetrievalSettings(
        candidate_k=int(raw_retrieval["candidate_k"]),
        default_top_k=int(raw_retrieval["default_top_k"]),
        min_top_k=int(raw_retrieval["min_top_k"]),
        max_top_k=int(raw_retrieval["max_top_k"]),
    )

    if retrieval.candidate_k <= 0:
        raise ValueError(
            "retrieval.candidate_k phải lớn hơn 0."
        )

    if retrieval.min_top_k <= 0:
        raise ValueError(
            "retrieval.min_top_k phải lớn hơn 0."
        )

    if retrieval.min_top_k > retrieval.max_top_k:
        raise ValueError(
            "min_top_k không được lớn hơn max_top_k."
        )

    if not (
        retrieval.min_top_k
        <= retrieval.default_top_k
        <= retrieval.max_top_k
    ):
        raise ValueError(
            "default_top_k phải nằm trong khoảng "
            "min_top_k và max_top_k."
        )

    raw_generation = _require_mapping(
        raw_config,
        "generation",
    )

    return ProductionSettings(
        default_mode=default_mode,
        modes=modes,
        retrieval=retrieval,
        default_include_explanation=bool(
            raw_generation.get(
                "default_include_explanation",
                True,
            )
        ),
    )