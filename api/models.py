from pydantic import BaseModel


class SourceRef(BaseModel):
    title: str
    url: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    query_type: str
