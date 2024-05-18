from fastapi import APIRouter

from ai.models.ocr_model import ocr_tag
from api.dto.ocr_dto import OcrRequest, OcrResponse

router = APIRouter(
    prefix="/ocr",
    tags=["OCR API"]
)

@router.post("/price-tag")
async def get_ocr_result(
        request_body: OcrRequest
):
    ocr_result = ocr_tag(request_body.imageUrl)

    return OcrResponse(
        brand=ocr_result["brand"],
        product=ocr_result["product"]
    )
