import streamlit as st
import json
import datetime
from PIL import Image
from testTools import frontTools

# debug or release, debug is only for testing
# debug
# FILL_IN_YOUR_API_KEY_HERE_BEFORE_ENTER_THE_SYSTEM_IF_CHOOSE_RELEASE_MODE
tools = frontTools(mode = 'release')
tools.setAPIKey(key = "sk-3b9c860b83844a789ea639881a569acc")
st.set_page_config(
    page_title="个性化健康宣教系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("🔐 登录系统")
    # password = st.text_input("请输入访问密码", type="password")
    # if password != "123456":
    #     st.warning("请输入正确密码访问系统。")
    #     st.stop()

lang = st.sidebar.selectbox("🌐 语言 / Language", ["中文", "English"])

def _(zh, en):
    return zh if lang == "中文" else en

st.title("🩺 " + _("个性化健康宣教系统", "Personalized Health Education System"))
st.markdown(_("请上传患者病历 JSON 文件，系统将自动生成宣教内容与随访建议。",
              "Upload the patient's JSON file to generate health education content and follow-up suggestions."))

with st.sidebar:
    uploaded_file = st.file_uploader(_("上传病历 JSON 文件", "Upload Patient JSON"), type="json")

if uploaded_file:
    try:
        data = json.load(uploaded_file)
        tools.prepareData(data)
        
        patients = tools.patients
        selected_id = st.selectbox(_("选择患者ID", "Select Patient ID"), list(patients.keys()))
        patient_records = patients[selected_id]

        with st.expander(_("📋 查看原始病历数据", "📋 View Raw Medical Records")):
            entry = patient_records[0]
            st.markdown(f"#### 📌 {_('标准化病历记录', 'Record')} ")
            st.json(entry)

        if st.button("🚀 " + _("生成宣教内容与随访建议", "Generate Health Education & Follow-up Plan")):
            with st.spinner(_("生成中，请稍候...", "Generating, please wait...")):
                tools.summary(entry)
                tools.advice(entry)
                tools.pipeline(entry)
                tools.generateVideo(entry)
                edu = {}
                video = tools.videoPath
                print(video)
                # '/root/autodl-fs/healthRAG/test_video_3.mp4'
                # 
                edu["text"] = tools.summaryText
                followup = {}
                followup["text"] = tools.followup
                followup["calendar"] = tools.calender

            st.success("✅ " + _("生成完成！请查看以下内容", "Generation completed!"))

            tabs = st.tabs([
                "📄 " + _("宣教说明", "Education"),
                "🖼 " + _("流程图", "Flowcharts"),
                "🎬 " + _("剧本", "Script"),
                "📆 " + _("随访建议", "Follow-up"),
                "📅 " + _("日历计划", "Calendar"),
                "📥 " + _("下载", "Download")
            ])

            with tabs[0]:
                st.markdown(edu["text"])

            with tabs[1]:
                image = Image.open(tools.pipelinePath)   
                st.image(image, use_column_width=True)

            with tabs[2]:
                st.markdown("### 🎞️ " + _("宣教视频", "Import Animation Videos"))
                if video:
                    st.video(video)
                else:
                    st.info(_("未找到可播放的视频文件。请将 MP4 等视频放入 videos 文件夹。", 
                                "No video files found. Please put MP4s in the videos folder."))

            with tabs[3]:
                st.markdown(followup["text"])

            with tabs[4]:
                st.dataframe(followup["calendar"], use_container_width=True)
                selected_date = st.date_input(_("选择一个日期以计划随访", "Pick a date for follow-up"), value=datetime.date.today())
                st.info(f"📌 {_('您选择的随访日期是', 'Selected follow-up date:')} {selected_date}")

            with tabs[5]:
                st.download_button(_("📥 下载宣教文本", "Download Education Text"),
                                   edu["text"], file_name="education.txt")
                st.download_button(_("📥 下载随访建议", "Download Follow-up Plan"),
                                   followup["text"], file_name="followup.txt")

    except Exception as e:
        st.error(f"❌ {_('读取文件失败', 'Failed to read file')}: {e}")
else:
    st.info(_("请先上传病历 JSON 文件", "Please upload a JSON file to begin."))
