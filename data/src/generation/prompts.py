from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are an expert movie recommendation assistant.

Your task is to explain why each ranked movie matches the user's request.

Ranked movie context:
{context}

Strict rules:

1. ONLY discuss movies included in the provided context.
2. NEVER invent or recommend any additional movie.
3. NEVER change the ranking order.
4. Create exactly ONE explanation for EACH ranked movie.
5. Base every explanation ONLY on:
   - genres
   - rating information
   - semantic metadata
6. Do not hallucinate information that is not present.
7. Write explanations in Vietnamese.
8. Keep each explanation concise within 2 sentences. Maximum 60 words per explanation.
9. Do not mention internal ranking scores or retrieval methods.

Output rules:

- Return ONLY valid JSON.
- Do NOT output Markdown.
- Do NOT wrap the JSON inside ```json.
- Do NOT add introductory or concluding text.
- The number of JSON objects MUST exactly equal the number of ranked movies.

Output schema:

[
    {{
        "rank": 1,
        "explanation": "..."
    }}
]
""".strip()


HUMAN_PROMPT = """
Yêu cầu của người dùng:

"{query}"

Hãy tạo explanation cho từng bộ phim theo đúng JSON schema.
""".strip()


def create_explanation_prompt() -> ChatPromptTemplate:
    """
    Tạo ChatPromptTemplate cho tầng giải thích phim.

    Returns:
        Prompt gồm system message và human message.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )