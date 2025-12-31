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

def human_screen_capture(resizeRatio:float=0.2) -> bytes:
    """Montor what the human is doing in PC screen."""

    with mss.mss() as sct:
        # 全画面キャプチャ
        monitor = sct.monitors[0]  # 0は全モニタ、1は主モニタ
        screenshot = sct.grab(monitor)
        # mssのScreenShotオブジェクトをPillowのImageオブジェクトに変換
        img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        new_size = (int(img.width * resizeRatio), int(img.height * resizeRatio)) # 幅と高さを50%にリサイズ
        _resampling = getattr(PIL.Image, 'Resampling', PIL.Image)
        img_resized = img.resize(new_size, _resampling.LANCZOS)
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

def decide_capture_mode(user_instruction: str, model: LiteLLMModel, system_prompt_prefix: str) -> str:
    """Decide capture mode based on the user's instruction.
    Returns one of: 'webcam_capture', 'screen_capture', 'alternate'."""
    prompt = (
        "あなたはシステム設定エージェントです。次の指示文から、"
        "人間の監視に最適なキャプチャモードを1つだけ選んでください。"
        "必ず以下のいずれかを返してください:\n"
        "  - webcam_capture(Webカメラ)\n"
        "  - screen_capture(画面のスクリーンショット)\n"
        "  - alternate(カメラと画面の交互)\n"
        "悩む場合は、'alternate'を選んでください。\n"
        "理由や余分なテキストは書かないでください。\n\n"
        f"指示文:\n{user_instruction}"
    )
    agent = Agent(model=litellm_model, system_prompt=prompt, messages=[Message(role='user', content=[{'text': user_instruction}])])
    try:
        res = agent()
        res_text = str(res).strip().lower()
    except Exception as e:
        print("Warning: mode decision agent failed:", e)
        res_text = ""
    if "webcam" in res_text or "カメラ" in res_text:
        return "webcam_capture"
    if "screen" in res_text or "スクリーン" in res_text:
        return "screen_capture"
    if "alternate" in res_text or "交互" in res_text:
        return "alternate"
    # fallback to interactive prompt
    return "alternate"

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

mode = decide_capture_mode(userorder, litellm_model, systemp)
print("選択されたキャプチャモード:", mode)

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
    # capture according to selected mode
    try:
        if mode == "webcam_capture":
            source_bytes = human_webcam_capture()
            source_str:str = "webcam"
        elif mode == "screen_capture":
            source_bytes = human_screen_capture()
            source_str:str = "screen"
        else:  # alternate
            if loopcnt % 2 != 0:
                source_bytes = human_webcam_capture()
                source_str:str = "webcam"
            else:
                source_bytes = human_screen_capture()
                source_str:str = "screen"
    except Exception as e:
        print("Warning: capture failed, falling back to screen capture:", e)
        source_bytes = human_screen_capture()

    user_msg = Message(
        role="user",
        content=[
            {
                "text": f"""{ get_current_datetime() }現在の{ source_str }状況です。"""
            },
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": source_bytes}
                }
            }
        ]
    )
    messages.append(user_msg)

    agent = Agent(model=litellm_model, system_prompt=systemp, messages=messages, tools=[speak])
    result = agent()
    previous_status = str(result)

    # messagesは参照渡しなので、すでに更新されていた
    #messages.append(Message(role="assistant", content=[{"text": previous_status}]))
    #print("\n----- previous_status -----")
    #print(previous_status)
    newmessages = []
    for m in messages[-5:]:
        newrole=m.get("role","unknown")
        newcontent=m.get("content", [{ "text": "unkowntext1" }])
        newcont0=newcontent[0]
        newmessages.append(Message(role=newrole, content=[newcont0]))
    messages = newmessages
    import pprint
    print("---- messages ----")
    pprint.pprint(messages)