import streamlit as st
from openai import OpenAI
import time
import re
from sentiment import predict_sentiment

placeholderstr = "Please input your command"
user_name = "Andrew"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def main():
    st.set_page_config( # 設定Web UI
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar: # with st.sidebar 表接下來的設定針對sidebar
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True) # sidebar中多加一個框框
        with st_c_1: # 接下來的設定針對剛創建的匡匡
            st.image("https://www.w3schools.com/howto/img_avatar.png") # 大頭貼

    st_c_chat = st.container(border=True) # 建立一個聊天區塊的 UI 容器，st.container() 會在頁面上建立一塊區域。border=True 表示這個區塊會有邊框，方便視覺辨識。

    if "messages" not in st.session_state: # 檢查過去紀錄中是否有messages
        st.session_state.messages = [] # 若無，就創建空的字串，供使用者後續輸入messages時可以存儲messages
    else:
        for msg in st.session_state.messages: # 若在過去紀錄中有找到messages，就把每則訊息依序讀取。 messages is a list of dicts，如同[{'role':..., 'content':...}, ....]
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"])) # 顯示一個聊天泡泡（chat_message），並附上頭像 avatar=user_image
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def generate_response(prompt):
        pattern = r'\b(i(\'?m| am| feel| think i(\'?)?m)?\s*(so\s+)?(stupid|ugly|dumb|idiot|worthless|loser|useless))\b'
        if re.search(pattern, prompt, re.IGNORECASE):
            return "Yes, you are!"
        else:
            return f"You say: {prompt}."
        
    # Chat function section (timing included inside function)
    def chat(prompt: str): 
        st_c_chat.chat_message("user",avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        # response = f"You type: {prompt}"

        predicted_sentiment = predict_sentiment(prompt)
        response2 = f"\n這則評論情緒為：**{predicted_sentiment}** 😊"

        st.session_state.messages.append({"role": "assistant", "content": response + response2})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response + response2))

    
    # := 等同於先assign並檢查：先將使用者輸入的內容assign給prompt，再看prompt這個變數為True or False
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"): # st.chat_input() returns 使用者輸入的內容（str），或是 None（如果還沒輸入）
        chat(prompt)  # 讓使用者輸入文字後觸發 chat() 函式

if __name__ == "__main__": 
    main()
# 就是在告訴 Python： 💬「如果這個 .py 檔是被直接執行的，就執行 main() 函數；但如果它只是被別人 import，就不要執行 main()。」

# 每個 .py 檔案都有一個內建變數 __name__
# 若該檔案是「直接執行的腳本」，則 __name__ 會自動被設為 "__main__"
# 若該檔案是「被別的 Python 檔案 import 進來」，則 __name__ 會是模組的名字
