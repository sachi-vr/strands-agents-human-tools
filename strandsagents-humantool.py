import cv2
import os
import io
import PIL.Image
import lmstudio as lms
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.types.content import Message
import time
from strands import tool
import mss

def get_current_datetime() -> str:
    """Get the current date and time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def human_screen_capture(resizeRatio:float=1.0) -> bytes:
    """Montor what the human is doing in PC screen."""

    with mss.mss() as sct:
        # 全画面キャプチャ
        monitor = sct.monitors[0]  # 全画面のモニター情報を取得
        screenshot = sct.grab(monitor)
        # mssのScreenShotオブジェクトをPillowのImageオブジェクトに変換
        img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        new_size = (int(img.width * resizeRatio), int(img.height * resizeRatio)) # 幅と高さを50%にリサイズ
        img_resized = img.resize(new_size, PIL.Image.LANCZOS)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='PNG', optimize=True, compress_level=9)
        # Reset the buffer position to the beginning
        buffer.seek(0)
        return buffer.getvalue()


def human_webcam_capture() ->  bytes:
    """Monitor the human facial expressions with a webcam."""

    # Open the webcam
    cap = cv2.VideoCapture(0)
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Release the webcam
    cap.release()
    # Convert the frame to a PIL Image
    img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True, compress_level=9)
    # Reset the buffer position to the beginning
    buffer.seek(0)
    return buffer.getvalue()

import pyttsx3
@tool
def speak(text:str) -> bool:
    """Speak a text string using the text-to-speech engine. The human can hear it."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return True

## ここから下は実行部分 ##
import logging
#logging.getLogger("strands").setLevel(logging.DEBUG)
#logging.basicConfig(
#    format="%(levelname)s | %(name)s | %(message)s", 
#    handlers=[logging.StreamHandler()]
#)

# pythonのlmstudioライブラリでモデルのロード
modelstr:str="qwen/qwen3-vl-8b"
model = lms.llm(modelstr, ttl=180)

all_loaded_models = lms.list_loaded_models()
print("Loaded models:", all_loaded_models)

# LiteLLM用のLM Studio設定の環境変数
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"  # 任意の文字列（LM Studioは無視）

# LM Studio の OpenAI 互換エンドポイントを指定
# モデル名にlm_studio/が付く
litellm_model = LiteLLMModel(model_id="lm_studio/"+modelstr)

userorder:str=input("指示を入力:").strip()
print("----")
systemp:str="指示: "+userorder
print(systemp)
previous_status="今の状況: 開始状態"
loopcnt=0

# Initialize messages once so history is preserved; exclude image elements
messages: list[Message] = [
    Message(
        role="assistant",
        content=[{ "text": f"{previous_status}" }]
    )
]

while True:
    loopcnt += 1
    print("\n---- loop "+str(loopcnt)+" -----")

    # messagesに含まれる、画像要素を削除しておく
    for m in messages:
        if hasattr(m, "content") and isinstance(m.content, list): # type: ignore
            m.content = [c for c in m.content if not (isinstance(c, dict) and "image" in c)] # type: ignore
    import pprint
    print("---- messages ----")
    pprint.pprint(messages)

    user_msg = Message(
        role="user",
        content=[
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": human_webcam_capture()}
                }
            },
            {
                "text": f"""{ get_current_datetime() }現在の状況です。"""
            }
        ]
    )
    messages.append(user_msg)

    agent = Agent(model=litellm_model, system_prompt=systemp, messages=messages, tools=[speak])
    result = agent()
    previous_status = str(result)

    # Append assistant reply to history
    messages.append(Message(role="assistant", content=[{"text": previous_status}]))
    #print("\n----- previous_status -----")
    #print(previous_status)