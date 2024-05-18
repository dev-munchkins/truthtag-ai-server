from pydantic import BaseModel


class OcrRequest(BaseModel):
    imageUrl: str


class OcrResponse(BaseModel):
    brand: str
    product: str
