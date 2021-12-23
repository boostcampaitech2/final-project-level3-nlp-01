from util import translate


def messageTemplate(text: str, left: bool = True, translate: bool = False) -> str:
    bg_color = "bg-gray-300" if left else "bg-green-300"
    if translate:
        bg_color = "bg-yellow-300"
        text = "&#128038 " + text
    div_align = "" if left else "mr-0 ml-auto "
    message = f"""<div class="clearfix"><div class="{bg_color} {div_align}bg-gray-300 w-3/4 mx-4 my-2 p-2 rounded-lg">{text}</div></div>"""
    return message


def sendMessage(text: str, target_lang: str, left: bool = True) -> str:
    original_text = messageTemplate(text, left)
    translate_text = messageTemplate(translate(text, target_lang), left, True)
    return original_text + translate_text
