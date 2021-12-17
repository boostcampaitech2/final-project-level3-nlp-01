import streamlit as st

from confirm_button_hack import cache_on_button_press
from create_text import sendMessage

st.set_page_config(
    page_title="런앤런팀 앵무새톡",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)

root_password = "password"


st.sidebar.image("https://i.imgur.com/145udIs.png")
password = st.sidebar.text_input("password", type="password")


@cache_on_button_press("Authenticate")
def authenticate(password) -> bool:
    print(type(password))
    return password == root_password


@st.cache(allow_output_mutation=True)
def messageBox():
    return []


def main():
    messages = messageBox()
    col1, col2 = st.columns(2)
    col1.subheader("User1")
    lang1 = col1.selectbox(
        "Select language", ("한국어", "english", "中文"), key="user1_lang"
    )
    sentence1 = col1.text_input("Message", key="user1_msg")
    btn1 = col1.button("Send", key="send1")

    col1.subheader("User2")
    lang2 = col1.selectbox(
        "Select language", ("한국어", "english", "中文"), index=1, key="user2_lang"
    )
    sentence2 = col1.text_input("Message", key="user2_msg")
    btn2 = col1.button("Send", key="send2")

    message_container = col2.container()
    message_container.write("")
    message_container.markdown(
        '<link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    if btn1:
        messages.append(sendMessage(sentence1, lang2, True))
    elif btn2:
        messages.append(sendMessage(sentence2, lang1, False))
    message_container.markdown(
        "".join(messages),
        unsafe_allow_html=True,
    )


if authenticate(password):
    st.sidebar.success("You are authenticated!")
    main()
else:
    st.sidebar.error("The password is invalid.")
