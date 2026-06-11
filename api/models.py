from pydantic import BaseModel, field_validator


class SourceRef(BaseModel):
    title: str
    url: str


class AskRequest(BaseModel):
    question: str

    @field_validator("question", mode="before")
    @classmethod
    def reject_blank(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        if len(stripped) > 500:
            raise ValueError("question must be 500 characters or fewer")
        return stripped


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    query_type: str


class HealthResponse(BaseModel):
    status: str
    collection_size: int
