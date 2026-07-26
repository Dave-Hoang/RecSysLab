import os
import json

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    ENV_PATH,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY_ENV_NAME,
    LLM_TEMPERATURE,
)
from src.generation.prompts import create_explanation_prompt


_explanation_chain = None


REQUIRED_RANKED_COLUMNS = {
    "final_rank",
    "title",
    "genres",
    "rating_mean",
    "rating_count",
    "page_content",
}


def _validate_ranked_movies(
    ranked_movies: pd.DataFrame,
) -> None:
    """
    Kiểm tra DataFrame ranking có đủ thông tin cho LLM hay không.

    Raises:
        TypeError:
            Nếu đầu vào không phải DataFrame.
        ValueError:
            Nếu thiếu cột bắt buộc.
    """
    if not isinstance(ranked_movies, pd.DataFrame):
        raise TypeError(
            "ranked_movies phải là một pandas DataFrame."
        )

    missing_columns = REQUIRED_RANKED_COLUMNS.difference(
        ranked_movies.columns
    )

    if missing_columns:
        raise ValueError(
            "Ranked movies DataFrame thiếu các cột: "
            f"{sorted(missing_columns)}"
        )


def _safe_float(value: object, default: float = 0.0) -> float:
    """
    Chuyển giá trị thành float an toàn.
    """
    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    """
    Chuyển giá trị thành int an toàn.
    """
    if value is None or pd.isna(value):
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_movies_for_llm(
    ranked_movies: pd.DataFrame,
) -> str:
    """
    Chuyển Top-N ranked movies thành context text sạch cho LLM.

    Không truyền các score nội bộ như:
    - semantic_similarity;
    - popularity_score;
    - rule_score;
    - cross_encoder_score;
    - final_score.

    Những score này phục vụ ranking, không cần thiết cho explanation.

    Args:
        ranked_movies:
            DataFrame đã được tầng ranking sắp xếp.

    Returns:
        Chuỗi context để truyền vào prompt.
    """
    if ranked_movies.empty:
        return ""

    _validate_ranked_movies(ranked_movies)

    context_blocks: list[str] = []

    ordered_movies = ranked_movies.sort_values(
        by="final_rank",
        ascending=True,
        kind="stable",
    )

    for row in ordered_movies.itertuples(index=False):
        rank = _safe_int(
            getattr(row, "final_rank", 0)
        )

        title = str(
            getattr(row, "title", "Không rõ tên phim")
        ).strip()

        genres = str(
            getattr(row, "genres", "N/A")
        ).strip()

        rating_mean = _safe_float(
            getattr(row, "rating_mean", 0.0)
        )

        rating_count = _safe_int(
            getattr(row, "rating_count", 0)
        )

        page_content = str(
            getattr(row, "page_content", "")
        ).strip()

        movie_block = (
            f"Rank {rank}\n"
            f"Title: {title}\n"
            f"Genres: {genres}\n"
            f"Rating: {rating_mean:.2f}/5.00 "
            f"from {rating_count:,} ratings\n"
            f"Semantic metadata:\n{page_content}"
        )

        context_blocks.append(movie_block)

    return "\n\n---\n\n".join(context_blocks)


def load_google_api_key() -> str:
    """
    Load GOOGLE_API_KEY từ file .env.

    Returns:
        API key.

    Raises:
        ValueError:
            Nếu không tìm thấy API key.
    """
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )

    api_key = os.getenv(GOOGLE_API_KEY_ENV_NAME)

    if not api_key:
        raise ValueError(
            f"Không tìm thấy {GOOGLE_API_KEY_ENV_NAME}.\n"
            f"Hãy kiểm tra file: {ENV_PATH}"
        )

    return api_key


def build_explanation_chain():
    """
    Khởi tạo LCEL explanation chain theo cơ chế singleton.

    Pipeline:

        ChatPromptTemplate
        → Gemini
        → StrOutputParser

    Returns:
        Runnable LCEL chain.
    """
    global _explanation_chain

    if _explanation_chain is not None:
        return _explanation_chain

    api_key = load_google_api_key()

    print(f"[*] LLM model: {GEMINI_MODEL_NAME}")
    print(f"[*] Temperature: {LLM_TEMPERATURE}")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        google_api_key=api_key,
    )

    prompt = create_explanation_prompt()

    print("=" * 80)
    print(prompt.input_variables)
    print("=" * 80)
    print(prompt.messages[0].prompt.template)
    
    _explanation_chain = (
        prompt | llm | StrOutputParser()
    )

    print("[✓] Explanation chain đã sẵn sàng.")

    return _explanation_chain


def explain_ranked_movies(
    query: str,
    ranked_movies: pd.DataFrame,
) -> list[str]:
    """
    Tạo lời giải thích cho danh sách phim đã được ranking.

    Hàm này KHÔNG:
    - retrieve;
    - load FAISS;
    - rerank;
    - thay đổi thứ tự phim.

    Args:
        query:
            Yêu cầu của người dùng.
        ranked_movies:
            Top-N DataFrame từ tầng ranking.

    Returns:
        Danh sách explanation tương ứng với từng movie.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    if ranked_movies.empty:
        return []

    context = format_movies_for_llm(ranked_movies)

    chain = build_explanation_chain()

    print(
        "[*] Đang gửi Top phim đã ranking tới Gemini "
        "để tạo explanation..."
    )

    response = chain.invoke(
        {
            "context": context,
            "query": cleaned_query,
        }
    )

    cleaned_response = str(response).strip()

    if not cleaned_response:
        raise RuntimeError(
            "LLM trả về nội dung rỗng."
        )

    # Gemini đôi khi bọc JSON trong ```json ... ```
    if cleaned_response.startswith("```"):
        cleaned_response = (
            cleaned_response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:
        parsed = json.loads(cleaned_response)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LLM không trả về JSON hợp lệ."
        ) from exc

    if not isinstance(parsed, list):
        raise RuntimeError(
            "LLM phải trả về một JSON array."
        )

    explanations: list[str] = []

    for item in parsed:
        if not isinstance(item, dict):
            raise RuntimeError(
                "Mỗi phần tử trong Json phải là object"
            )
        if "explanation" not in item:
            raise RuntimeError(
                "Thiếu trường 'explanation' trong phản hồi của LLM."
            )
        
        explanation = str(item["explanation"]).strip()

        explanations.append(explanation)

    if len(explanations) != len(ranked_movies):
        raise RuntimeError(
            "Số lượng explanation không khớp với số lượng movie đã ranking."
        ) 

    return explanations
