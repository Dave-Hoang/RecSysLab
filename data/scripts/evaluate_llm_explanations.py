from __future__ import annotations

import sys
import time
from pathlib import Path
from time import perf_counter

# Tự động nạp thư mục data/ vào sys.path để tránh lỗi ModuleNotFoundError: No module named 'src'
DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import pandas as pd
from tqdm import tqdm

from src.config import EVALUATION_METRICS_DIR
from src.evaluation.explanation_evaluator import (
    DEFAULT_JUDGE_MODEL,
    evaluate_single_explanation,
)
from src.evaluation.query_loader import load_evaluation_queries
from src.services.recommendation_service import RecommendationService


def save_progress(detailed_records: list[dict], total_start: float) -> None:
    """Lưu kết quả chi tiết và tổng hợp xuống đĩa ngay lập tức."""
    if not detailed_records:
        return

    EVALUATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = EVALUATION_METRICS_DIR / "explanation_evaluation_details.csv"
    overall_path = EVALUATION_METRICS_DIR / "explanation_overall_metrics.csv"

    details_df = pd.DataFrame(detailed_records)
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    total_count = len(details_df)
    mean_faithfulness = details_df["faithfulness_score"].mean()
    mean_context_relevance = details_df["context_relevance_score"].mean()
    hallucinated_count = details_df["is_hallucinated"].sum()
    hallucination_rate = (
        (hallucinated_count / total_count) * 100.0 if total_count > 0 else 0.0
    )

    overall_df = pd.DataFrame(
        [
            {
                "judge_model": DEFAULT_JUDGE_MODEL,
                "total_explanations_evaluated": total_count,
                "mean_faithfulness_score": round(mean_faithfulness, 4),
                "mean_context_relevance_score": round(mean_context_relevance, 4),
                "hallucinated_explanations_count": int(hallucinated_count),
                "hallucination_rate_percent": round(hallucination_rate, 2),
            }
        ]
    )

    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")


def main() -> None:
    print("=" * 80)
    print("LLM EXPLANATION EVALUATION BENCHMARK")
    print("Metrics: Faithfulness, Context Relevance, Hallucination Rate")
    print(f"Judge Model: {DEFAULT_JUDGE_MODEL}")
    print("=" * 80)

    EVALUATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = EVALUATION_METRICS_DIR / "explanation_evaluation_details.csv"
    overall_path = EVALUATION_METRICS_DIR / "explanation_overall_metrics.csv"

    detailed_records: list[dict] = []
    completed_keys: set[tuple[str, int]] = set()

    # Nạp dữ liệu checkpoint nếu đã chạy trước đó
    if details_path.exists():
        try:
            existing_df = pd.read_csv(details_path)
            detailed_records = existing_df.to_dict(orient="records")
            for row in detailed_records:
                completed_keys.add((str(row.get("query_id")), int(row.get("movieId"))))
            print(
                f"[✓] Đã nạp checkpoint trước đó: {len(detailed_records)} câu giải thích từ {details_path.name}."
            )
        except Exception as exc:
            print(f"[!] Không thể đọc checkpoint cũ ({exc}), sẽ chạy mới hoàn toàn.")

    queries = load_evaluation_queries()
    print(f"[✓] Đã nạp {len(queries)} evaluation queries.")

    print("[*] Đang khởi tạo RecommendationService...")
    service = RecommendationService(top_n=5)

    total_start = perf_counter()

    # Khởi tạo tqdm progress bar cho danh sách queries
    pbar = tqdm(
        queries,
        desc="[Benchmark Progress]",
        unit="query",
        dynamic_ncols=True,
    )

    try:
        for query_idx, eval_query in enumerate(pbar, start=1):
            pbar.set_postfix(
                query_id=eval_query.query_id,
                total_records=len(detailed_records),
            )

            tqdm.write(
                f"\n[{query_idx}/{len(queries)}] Đang xử lý: {eval_query.query_id} - {eval_query.query!r}"
            )

            # Kiểm tra xem Query này đã hoàn thành đủ 5 phim trong checkpoint trước đó chưa
            existing_count = sum(1 for k in completed_keys if k[0] == str(eval_query.query_id))
            if existing_count >= 5:
                tqdm.write(
                    f"  [✓] Đã hoàn thành 5/5 phim cho {eval_query.query_id} trước đó. BỎ QUA hoàn toàn (Không gọi API Gemini)!"
                )
                continue

            try:
                res = service.recommend(
                    query=eval_query.query,
                    include_explanation=True,
                    configuration="hybrid_with_ce",
                    top_n=5,
                )
            except Exception as exc:
                tqdm.write(f"  [!] Lỗi khi chạy recommendation pipeline: {exc}")
                continue

            recommendations = res.recommendations
            query_updated = False

            for movie in recommendations:
                movie_id = int(movie.get("movieId", -1))
                if (str(eval_query.query_id), movie_id) in completed_keys:
                    tqdm.write(
                        f"  [✓] Bỏ qua phim: {movie.get('title')} (Đã chấm điểm trước đó)"
                    )
                    continue

                explanation = movie.get("explanation", "")
                if not explanation:
                    continue

                tqdm.write(
                    f"  -> Đang chấm điểm phim: {movie.get('title')} (Rank {movie.get('final_rank')})..."
                )

                try:
                    judge_result = evaluate_single_explanation(
                        query=eval_query.query,
                        movie=movie,
                        explanation=explanation,
                        judge_model_name=DEFAULT_JUDGE_MODEL,
                    )

                    record = {
                        "query_id": eval_query.query_id,
                        "query": eval_query.query,
                        "category": eval_query.category.value,
                        "difficulty": eval_query.difficulty.value,
                        "rank": movie.get("final_rank"),
                        "movieId": movie_id,
                        "title": movie.get("title"),
                        "genres": movie.get("genres"),
                        "explanation": explanation,
                        "claims": " | ".join(judge_result.claims),
                        "faithfulness_score": judge_result.faithfulness_score,
                        "faithfulness_reason": judge_result.faithfulness_reason,
                        "context_relevance_score": judge_result.context_relevance_score,
                        "context_relevance_reason": judge_result.context_relevance_reason,
                        "is_hallucinated": judge_result.is_hallucinated,
                    }

                    detailed_records.append(record)
                    completed_keys.add((str(eval_query.query_id), movie_id))
                    query_updated = True

                except Exception as exc:
                    tqdm.write(
                        f"     [!] Lỗi chấm điểm phim {movie.get('title')}: {exc}"
                    )

                # Rate Limiting delay giữa các API requests
                time.sleep(0.5)

            if query_updated:
                save_progress(detailed_records, total_start)
                pbar.set_postfix(
                    query_id=eval_query.query_id,
                    total_records=len(detailed_records),
                )

    except KeyboardInterrupt:
        tqdm.write("\n\n" + "=" * 80)
        tqdm.write("[!] ĐÃ TẠM DỪNG TIẾN TRÌNH THEO YÊU CẦU NGƯỜI DÙNG (CTRL+C)")
        save_progress(detailed_records, total_start)
        tqdm.write(
            f"[✓] Đã tự động lưu Checkpoint: {len(detailed_records)} kết quả đã đánh giá."
        )
        tqdm.write(
            "[✓] Lần sau khi chạy lại lệnh, script sẽ TỰ ĐỘNG NỐI TIẾP (Resume) từ câu chưa xong!"
        )
        tqdm.write("=" * 80)
        return

    if not detailed_records:
        print("\n[!] Không có bản ghi đánh giá nào được tạo thành công.")
        return

    save_progress(detailed_records, total_start)
    details_df = pd.DataFrame(detailed_records)

    total_count = len(details_df)
    mean_faithfulness = details_df["faithfulness_score"].mean()
    mean_context_relevance = details_df["context_relevance_score"].mean()
    hallucinated_count = details_df["is_hallucinated"].sum()
    hallucination_rate = (
        (hallucinated_count / total_count) * 100.0 if total_count > 0 else 0.0
    )

    total_time = perf_counter() - total_start

    print("\n" + "=" * 80)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ LLM EXPLANATION HOÀN TẤT")
    print("=" * 80)
    print(f"Tổng số lời giải thích đã đánh giá: {total_count}")
    print(f"Mean Faithfulness Score           : {mean_faithfulness:.4f} / 1.0")
    print(f"Mean Context Relevance Score      : {mean_context_relevance:.4f} / 1.0")
    print(f"Hallucinated Explanations Count   : {hallucinated_count} / {total_count}")
    print(f"Hallucination Rate (%)            : {hallucination_rate:.2f}%")
    print(f"\n[✓] Chi tiết lưu tại              : {details_path}")
    print(f"[✓] Tổng hợp lưu tại              : {overall_path}")
    print(f"[✓] Tổng thời gian thực thi        : {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
