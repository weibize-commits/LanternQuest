from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

LLMPurpose = Literal[
    "intent_structuring",
    "retrieval_query_proposal",
    "plan_generation",
    "grounded_explanation",
    "environment_action",
    "residual_repair",
    "candidate_ranking",
]


class LLMRequest(BaseModel):
    """Auditable request boundary for an interchangeable language model."""

    model_config = ConfigDict(extra="forbid")

    purpose: LLMPurpose
    system_instruction: str
    payload: dict[str, Any]
    response_schema: dict[str, Any]


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    content: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    normalization_applied: str | None = None
    provider_calls: int = 1


class LLMAdapterError(ValueError):
    """Adapter failure with auditable provider-call and token counts."""

    def __init__(
        self,
        message: str,
        *,
        provider_calls: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.provider_calls = provider_calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMAdapter(Protocol):
    """Provider-neutral LLM interface; provider calls are implemented separately."""

    model_id: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...
