import os
import uuid
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI
from moviepy.editor import (
    ImageClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, AudioFileClip
)
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tts.v20190823 import tts_client, models


class HealthVideoGenerator:
    FONT_REGULAR = "fonts/NotoSansSC-Regular.ttf"
    FONT_BOLD = "fonts/NotoSansSC-ExtraBold.ttf"

    SCENE_TEMPLATES = {
        "greeting": {"background": "hospital_room.png", "images": ["doctor_wave.png"]},
        "medication": {"background": "pharmacy.png", "images": ["pills.png"]},
        "diet": {"background": "kitchen.png", "images": ["vegetables.png"]},
        "exercise": {"background": "park.png", "images": ["running.png"]},
        "sleep": {"background": "bedroom.png", "images": ["sleep.png"]},
        "warning": {"background": "hospital_room.png", "images": ["warning.png"]},
        "thanks": {"background": "hospital_room.png", "images": ["doctor_wave.png"]}
    }

    def __init__(self, user_name, user_gender, health_content):
        load_dotenv()

        self.user_name = user_name
        self.user_gender = user_gender
        self.health_content = health_content

        self.assets_dir = "assets"
        self.output_dir = "./outputs"
        self.audio_dir = "audio"
        self.video_path = os.path.join(self.output_dir, "final_video.mp4")
        self.script_path = os.path.join(self.output_dir, "scene_script.json")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        cred = credential.Credential(
            os.getenv("TENCENT_SECRET_ID"),
            os.getenv("TENCENT_SECRET_KEY")
        )
        http_profile = HttpProfile(endpoint="tts.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        self.tts_client = tts_client.TtsClient(cred, "", client_profile)

    def generate_script(self):
        prompt = f"""
你是一个医学健康视频的编剧助手。根据以下健康宣教内容，生成一个视频脚本的场景列表（JSON格式），包含一个或多个你认为需要涵盖的场景。

要求：
1. 只生成中间场景（无需问候和感谢）
2. 每个场景的类型必须从以下中选择："medication", "diet", "exercise", "sleep", "warning"
3. 每个场景必须包含：
   - scene_type: 类型
   - text_label: 用于语音播报和字幕的完整句子
   - keyword: 提炼的需要用户注意的关键内容

健康宣教内容：
{self.health_content}

请只输出 JSON 数组，不要解释说明。
        """
        response = self.openai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return json.loads(response.choices[0].message.content)

    def assemble_scenes(self, generated_scenes):
        honorific = "先生" if self.user_gender == "男" else "女士"
        full_scenes = []

        def build_scene(scene_type, text_label, keyword):
            return {
                "scene_type": scene_type,
                "background": self.SCENE_TEMPLATES[scene_type]["background"],
                "background_path": os.path.join(self.assets_dir, "background", self.SCENE_TEMPLATES[scene_type]["background"]),
                "images": self.SCENE_TEMPLATES[scene_type]["images"],
                "image_paths": [os.path.join(self.assets_dir, "images", img) for img in self.SCENE_TEMPLATES[scene_type]["images"]],
                "text_label": text_label,
                "keyword": keyword
            }

        full_scenes.append(build_scene(
            "greeting",
            f"你好，{self.user_name}{honorific}，欢迎来到健康时间。",
            "欢迎来到健康时间"
        ))

        for item in generated_scenes:
            stype = item["scene_type"]
            if stype in self.SCENE_TEMPLATES:
                full_scenes.append(build_scene(stype, item["text_label"], item["keyword"]))

        full_scenes.append(build_scene(
            "thanks",
            f"{self.user_name}{honorific}，感谢您的收看，祝您健康平安。",
            "感谢"
        ))

        return full_scenes

    def synthesize_tts(self, text, filename):
        req = models.TextToVoiceRequest()
        params = {
            "Text": text,
            "SessionId": str(uuid.uuid4()),
            "ModelType": 1,
            "VoiceType": 301000,
            "Speed": 0,
            "Volume": 0,
            "SampleRate": 16000,
            "Codec": "mp3"
        }
        req.from_json_string(json.dumps(params))
        resp = self.tts_client.TextToVoice(req)
        audio_data = base64.b64decode(resp.Audio)
        with open(filename, "wb") as f:
            f.write(audio_data)
        return AudioFileClip(filename).duration

    def slide_in_from_left(self, image_clip, screen_width=1280, target_x_ratio=0.2, duration=5, vertical_offset=0):
        resized_clip = image_clip.resize(height=300).set_duration(duration)
        image_width = resized_clip.w
        target_x = int(screen_width * target_x_ratio)
        target_y = int(720 / 2 + vertical_offset)
        return resized_clip.set_position(lambda t: (
            int(-image_width + (target_x + image_width) * min(t / 1.5, 1)),
            target_y
        ))

    def keyword_text(self, keyword, duration):
        return TextClip(keyword, font=self.FONT_BOLD, fontsize=60, color="black").set_duration(duration).set_position("center")

    def static_text(self, text, duration, y_position=600):
        return TextClip(text, font=self.FONT_REGULAR, fontsize=30, color="black", method='label').set_duration(duration).set_position(("center", y_position))

    def create_scene_clip(self, scene, audio_path, duration):
        bg = ImageClip(scene["background_path"]).set_duration(duration).resize(height=720)
        screen_width = bg.w

        img_layers = []
        for img_path in scene["image_paths"]:
            if os.path.exists(img_path):
                base_img = ImageClip(img_path).set_duration(duration)
                sliding_img = self.slide_in_from_left(base_img, screen_width, duration=duration)
                img_layers.append(sliding_img)

        keyword_clip = self.keyword_text(scene["keyword"], duration)
        subtitle_clip = self.static_text(scene["text_label"], duration, y_position=bg.h - 60)
        audio = AudioFileClip(audio_path)

        return CompositeVideoClip([bg, *img_layers, keyword_clip, subtitle_clip]).set_audio(audio).set_duration(duration)

    def save_script_json(self, scenes):
        with open(self.script_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)

    def generate_video(self, scenes):
        clips = []
        for idx, scene in enumerate(scenes):
            audio_path = os.path.join(self.audio_dir, f"scene{idx+1}.mp3")
            duration = self.synthesize_tts(scene["text_label"], audio_path)
            scene_clip = self.create_scene_clip(scene, audio_path, duration)
            clips.append(scene_clip)

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            self.video_path, fps=24, codec="libx264", audio_codec="libmp3lame",
            temp_audiofile="temp-audio.mp3", remove_temp=True
        )

    def run(self):
        print("🎬 开始生成视频")
        generated_script = self.generate_script()
        scenes = self.assemble_scenes(generated_script)
        self.save_script_json(scenes)
        self.generate_video(scenes)
        print(f"✅ 视频生成完成：{self.video_path}")
        print(f"✅ 脚本已保存：{self.script_path}")


# === 使用方式 ===
if __name__ == "__main__":
    generator = HealthVideoGenerator(
        user_name="李",
        user_gender="男",
        health_content=(
            "请注意避免食用海鲜、豆制品及动物内脏等食物，保持良好作息少熬夜。"
            "每日服用消炎药两次直至症状缓解，可以进行游泳等适度的运动，"
            "不要激烈运动，千万不要自行按摩或者热敷伤部。"
        )
    )
    generator.run()
