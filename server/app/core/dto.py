from pydantic import BaseModel


class ItemDto(BaseModel):
    text: str
    target_lang: str
