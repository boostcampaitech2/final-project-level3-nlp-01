from pydantic import BaseModel


class ItemDto(BaseModel):
    text: str
    source_lang: str
    target_lang: str
