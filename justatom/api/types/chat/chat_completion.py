from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ChatCompletionContentPartParam(BaseModel):
    type: Literal["text", "image_url"]
    text: str = None
    image_url: dict = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatCompletionContentPartParam]]


class ParagraphMessage(BaseModel):
    content: Union[str, List[ChatCompletionContentPartParam]]
    title: Union[str, List[ChatCompletionContentPartParam]] = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default="mlx-community/Qwen2.5-14B-Instruct-bf16")
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=1024)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)
    tools: Optional[List[Function]] = Field(default=None)
    tool_choice: Optional[str] = Field(default=None)


class ChatCompletionKeywordsRequest(BaseModel):
    model: Optional[str] = Field(default="mlx-community/Qwen2.5-14B-Instruct-bf16")
    messages: List[ParagraphMessage]
    source_language: Optional[str] = "русском"
    max_tokens: Optional[int] = Field(default=1024)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.228)
    tools: Optional[List[Function]] = Field(default=None)
    tool_choice: Optional[str] = Field(default=None)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionKeywordsResponse(BaseModel):
    created: int
    id: str
    choices: List[dict]


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
