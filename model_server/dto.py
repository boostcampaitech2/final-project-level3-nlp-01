from pydantic import BaseModel


class ItemDto(BaseModel):
    text: str
