import streamlit as st
import whisper
import os
import subprocess
import shutil
import tempfile
from datetime import timedelta, datetime
from openai import OpenAI
import io

# 페이지 설정
st.set_page_config(
    page_title="AI 자막 생성기",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def format_timestamp(seconds):
    """초를 SRT 형식의 타임스탬프로 변환"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

@st.cache_resource
def load_whisper_model(model_size):
    """Whisper 모델 로딩 (캐시 사용)"""
    return whisper.load_model(model_size)

def translate_srt_to_language(srt_content, openai_api_key, target_language="한국어"):
    """SRT 내용을 지정된 언어로 번역"""
    try:
        client = OpenAI(api_key=openai_api_key)
        
        # SRT 내용을 배치로 나누기
        max_chars_per_batch = 8000
        srt_lines = srt_content.strip().split('\n\n')
        batches = []
        current_batch = []
        current_length = 0
        
        for subtitle_block in srt_lines:
            block_length = len(subtitle_block)
            if current_length + block_length > max_chars_per_batch and current_batch:
                batches.append('\n\n'.join(current_batch))
                current_batch = [subtitle_block]
                current_length = block_length
            else:
                current_batch.append(subtitle_block)
                current_length += block_length
        
        if current_batch:
            batches.append('\n\n'.join(current_batch))
        
        # 각 배치를 번역
        translated_batches = []
        progress_bar = st.progress(0)
        
        for i, batch_text in enumerate(batches):
            st.write(f"📝 배치 {i+1}/{len(batches)} 번역 중...")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""당신은 전문 SRT 자막 번역가입니다. 다음 지침을 정확히 따라 {target_language}로 번역해주세요:
**형식 유지:**
- 자막 번호와 타임코드(00:00:00,000 --> 00:00:00,000)는 절대 변경하지 마세요
- SRT 형식과 빈 줄을 정확히 유지하세요
- 텍스트 부분만 번역하세요
**번역 원칙:**
- SRT는 시간 단위로 나뉘어져 문장이 중간에 끊어집니다
- 전후 맥락을 파악해서 자연스러운 완전한 문장으로 번역하세요
- 문장이 다음 자막으로 이어질 때는 자연스럽게 연결해서 번역하세요
- 중복되는 내용은 제거하고 매끄럽게 이어지도록 하세요
**언어 스타일:**
- 틱톡/SNS 영상용이므로 친근하고 자연스러운 구어체 사용
- 직역보다는 의역으로 {target_language} 화자가 실제 말하는 방식으로 번역
- 감탄사, 어미 등을 활용해 생동감 있게 표현
- 전문용어나 브랜드명은 현지화된 표현 사용
**품질 기준:**
- {target_language} 원어민이 듣기에 자연스러워야 함
- 원본의 톤과 감정을 그대로 전달
- 문화적 차이가 있는 표현은 현지 상황에 맞게 조정"""
                    },
                    {
                        "role": "user",
                        "content": f"다음 SRT 자막을 {target_language}로 번역해주세요:\n\n{batch_text}"
                    }
                ],
                max_tokens=10000,
                temperature=0.3
            )
            
            translated_batches.append(response.choices[0].message.content)
            progress_bar.progress((i + 1) / len(batches))
        
        return '\n\n'.join(translated_batches)
        
    except Exception as e:
        st.error(f"번역 중 오류 발생: {str(e)}")
        return None

def render_video_with_subtitles(video_file, subtitle_content, font_size=8, margin_bottom=50):
    """영상에 자막 렌더링"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 임시 파일들 생성
            video_path = os.path.join(temp_dir, "input_video.mp4")
            subtitle_path = os.path.join(temp_dir, "subtitle.srt")
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # 영상 파일 저장
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            
            # 자막 파일 저장
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write(subtitle_content)
            
            # FFmpeg 명령어 (자막 위치 조정 포함)
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"subtitles='{subtitle_path}':force_style='FontSize={font_size},PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=1,Shadow=1,MarginV={margin_bottom}'",
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            # FFmpeg 실행
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 렌더링된 영상 읽기
                with open(output_path, "rb") as f:
                    return f.read()
            else:
                st.error(f"FFmpeg 오류: {result.stderr}")
                return None
                
    except Exception as e:
        st.error(f"렌더링 중 오류 발생: {str(e)}")
        return None

def main():
    # 제목 및 설명
    st.title("🎬 AI 자막 생성기")
    # st.markdown("**Whisper AI**로 자막 추출 → **OpenAI GPT**로 번역 → **FFmpeg**로 렌더링")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # OpenAI API 키
        openai_api_key = st.text_input(
            "🔑 OpenAI API Key", 
            type="password",
            help="번역 기능을 사용하려면 OpenAI API 키가 필요합니다."
        )
        
        # Whisper 모델 선택
        model_size = st.selectbox(
            "🎤 Whisper 모델",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="큰 모델일수록 정확하지만 느립니다."
        )
        
        # 음성 인식 언어
        whisper_language = st.selectbox(
            "🗣️ 음성 인식 언어",
            [
                ("자동 감지", None),
                ("한국어", "ko"),
                ("영어", "en"),
                ("일본어", "ja"),
                ("중국어", "zh"),
                ("스페인어", "es"),
                ("프랑스어", "fr"),
                ("독일어", "de")
            ],
            format_func=lambda x: x[0]
        )
        
        # 번역 언어
        translate_language = st.selectbox(
            "🌍 번역 언어",
            [
                "번역 안함",
                "한국어",
                "영어", 
                "일본어",
                "중국어",
                "스페인어",
                "프랑스어",
                "독일어",
                "러시아어",
                "포르투갈어",
                "이탈리아어"
            ]
        )
        
        # 자막 렌더링 설정
        st.subheader("🎬 렌더링 설정")
        render_subtitles = st.checkbox("영상에 자막 렌더링", value=True)
        font_size = st.slider("자막 크기", 6, 20, 8)
        subtitle_margin = st.slider("자막 여백 (아래에서부터)", 20, 150, 50, help="숫자가 클수록 자막이 위로 올라갑니다")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 영상 업로드")
        uploaded_file = st.file_uploader(
            "영상 파일을 선택하세요",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="지원 형식: MP4, AVI, MOV, MKV, WebM"
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            # 처리 시작 버튼
            if st.button("🚀 자막 생성 시작", type="primary"):
                process_video(
                    uploaded_file, 
                    model_size, 
                    whisper_language[1], 
                    translate_language if translate_language != "번역 안함" else None,
                    openai_api_key,
                    render_subtitles,
                    font_size,
                    subtitle_margin
                )
    
    with col2:
        st.header("📋 결과")
        if "results" not in st.session_state:
            st.info("영상을 업로드하고 '자막 생성 시작' 버튼을 클릭하세요.")

def process_video(uploaded_file, model_size, whisper_language, translate_language, openai_api_key, render_subtitles, font_size, subtitle_margin):
    """영상 처리 메인 함수"""
    
    # 진행 상태 표시
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 처리 중...")
        
        # 1단계: Whisper 모델 로딩
        st.write("🎤 Whisper 모델 로딩 중...")
        model = load_whisper_model(model_size)
        st.success("✅ 모델 로딩 완료")
        
        # 2단계: 음성 인식
        st.write("🎵 음성 인식 진행 중...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        try:
            # Whisper로 음성 인식
            if whisper_language:
                result = model.transcribe(temp_file_path, language=whisper_language, word_timestamps=True)
            else:
                result = model.transcribe(temp_file_path, word_timestamps=True)
            
            # SRT 내용 생성
            srt_content = ""
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            
            st.success(f"✅ 음성 인식 완료 ({len(result['segments'])}개 자막 생성)")
            
            # 원본 SRT 다운로드 버튼
            st.download_button(
                "📥 원본 SRT 다운로드",
                srt_content,
                file_name=f"subtitle_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                mime="text/plain"
            )
            
            translated_content = srt_content
            
            # 3단계: 번역 (선택사항)
            if translate_language and openai_api_key:
                st.write(f"🌍 {translate_language}로 번역 중...")
                translated_content = translate_srt_to_language(srt_content, openai_api_key, translate_language)
                
                if translated_content:
                    st.success(f"✅ {translate_language} 번역 완료")
                    
                    # 번역된 SRT 다운로드 버튼
                    language_suffix = translate_language.lower().replace(" ", "_")
                    st.download_button(
                        f"📥 {translate_language} SRT 다운로드",
                        translated_content,
                        file_name=f"subtitle_{language_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                        mime="text/plain"
                    )
                else:
                    st.error("번역에 실패했습니다.")
                    translated_content = srt_content
            elif translate_language and not openai_api_key:
                st.warning("번역을 위해서는 OpenAI API 키가 필요합니다.")
            
            # 4단계: 영상 렌더링 (선택사항)
            if render_subtitles:
                st.write("🎬 영상에 자막 렌더링 중...")
                
                # 파일 포인터를 처음으로 리셋
                uploaded_file.seek(0)
                rendered_video = render_video_with_subtitles(uploaded_file, translated_content, font_size, subtitle_margin)
                
                if rendered_video:
                    st.success("✅ 자막 렌더링 완료")
                    
                    # 렌더링된 영상 다운로드 버튼
                    st.download_button(
                        "📥 자막 포함 영상 다운로드",
                        rendered_video,
                        file_name=f"video_with_subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
                    
                    # 미리보기
                    st.video(rendered_video)
                else:
                    st.error("영상 렌더링에 실패했습니다.")
            
            # 자막 내용 미리보기
            with st.expander("📄 자막 내용 미리보기"):
                if translate_language and translated_content != srt_content:
                    tab1, tab2 = st.tabs(["원본", f"{translate_language} 번역"])
                    with tab1:
                        st.text_area("원본 자막", srt_content, height=300)
                    with tab2:
                        st.text_area(f"{translate_language} 자막", translated_content, height=300)
                else:
                    st.text_area("자막 내용", srt_content, height=300)
            
        finally:
            # 임시 파일 삭제
            os.unlink(temp_file_path)
        
        st.balloons()
        st.success("🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
