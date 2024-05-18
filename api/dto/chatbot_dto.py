from pydantic import BaseModel


class ChatbotRequest(BaseModel):
    message: str


class ChatbotResponse(BaseModel):
    product: str
    function: int
