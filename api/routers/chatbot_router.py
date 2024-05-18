from fastapi import APIRouter

from ai.models.chatbot_model import chatbot
from api.dto.chatbot_dto import ChatbotRequest, ChatbotResponse

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot API"]
)

@router.post("")
async def get_chatbot_result(
        request_body: ChatbotRequest
):
    chatbot_result = chatbot(request_body.message)

    return ChatbotResponse(
        product=chatbot_result["product"],
        function=chatbot_result["function"]
    )
