from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_TEMPERATURE
from src.evaluation.explanation_judge_prompts import create_judge_prompt
from src.evaluation.explanation_judge_schema import ExplanationJudgeResult
from src.generation.explanation_chain import load_google_api_key

DEFAULT_JUDGE_MODEL = "gemini-3.6-flash"

_judge_chain = None


def clean_json_markdown(text: str) -> str:
    """
    Làm sạch chuỗi JSON nếu Gemini bọc trong ```json ... ```.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned.replace("```json", "")
            .replace("```", "")
            .strip()
        )

    return cleaned


def build_judge_chain(
    model_name: str = DEFAULT_JUDGE_MODEL,
):
    """
    Khởi tạo LCEL Judge Chain với temperature = 0.0.
    """
    global _judge_chain

    if _judge_chain is not None:
        return _judge_chain

    api_key = load_google_api_key()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=api_key,
    )

    prompt = create_judge_prompt()

    _judge_chain = prompt | llm | StrOutputParser()

    return _judge_chain


def format_movie_context_for_judge(
    movie: dict[str, Any],
) -> str:
    """
    Định dạng đầy đủ thông tin phim (Rank, Score, Metadata) cho Judge.
    """
    rank = movie.get("rank", movie.get("final_rank", "N/A"))
    score = movie.get("evaluation_score", movie.get("final_score", "N/A"))
    title = movie.get("title", "Unknown")
    genres = movie.get("genres", "N/A")
    rating_mean = movie.get("rating_mean", 0.0)
    rating_count = movie.get("rating_count", 0)
    page_content = movie.get("page_content", "")

    return (
        f"Rank Position: {rank}\n"
        f"Final Ranking Score: {score}\n"
        f"Title: {title}\n"
        f"Genres: {genres}\n"
        f"Rating: {rating_mean:.2f}/5.00 from {rating_count:,} ratings\n"
        f"Semantic Content Metadata:\n{page_content}"
    )


def evaluate_single_explanation(
    query: str,
    movie: dict[str, Any],
    explanation: str,
    judge_model_name: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 4,
    initial_delay: float = 2.0,
) -> ExplanationJudgeResult:
    """
    Chấm điểm 1 câu giải thích bằng Gemini LLM Judge với cơ chế Exponential Backoff Retry.
    """
    cleaned_query = query.strip()
    cleaned_explanation = explanation.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    if not cleaned_explanation:
        raise ValueError("Explanation không được để trống.")

    movie_context = format_movie_context_for_judge(movie)

    chain = build_judge_chain(model_name=judge_model_name)

    payload = {
        "query": cleaned_query,
        "movie_context": movie_context,
        "explanation": cleaned_explanation,
    }

    last_exception: Exception | None = None
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = chain.invoke(payload)
            cleaned_json_text = clean_json_markdown(str(raw_response))
            parsed_dict = json.loads(cleaned_json_text)

            return ExplanationJudgeResult.model_validate(parsed_dict)

        except Exception as exc:
            last_exception = exc

            if attempt < max_retries:
                print(
                    f"  [!] Lỗi khi gọi LLM Judge (Lần {attempt}/{max_retries}): {exc}. "
                    f"Thử lại sau {delay:.1f}s (Exponential Backoff)..."
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                break

    raise RuntimeError(
        f"Không thể hoàn thành LLM Judge evaluation sau {max_retries} lần thử."
    ) from last_exception
