from flask import Flask, render_template, request
from dotenv import load_dotenv
from groq import Groq
import os
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
import tempfile

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)

# ✅ If you are on Windows, uncomment and set this path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"




@app.route("/", methods=["GET", "POST"])
def index():
    mcqs = ""

    if request.method == "POST":
        difficulty = request.form.get("difficulty")
        count = request.form.get("count")
        count = int(count) if count else 5

        topic = request.form.get("topic", "").strip()

        pdf_file = request.files.get("pdf_file")
        txt_file = request.files.get("txt_file")
        docx_file = request.files.get("docx_file")
        image_file = request.files.get("image_file")
        audio_file = request.files.get("audio_file")
        video_file = request.files.get("video_file")

        extracted_text = ""
        source_used = "Topic"

        # ✅ TXT
        if txt_file and txt_file.filename != "":
            extracted_text = txt_file.read().decode("utf-8", errors="ignore")
            source_used = "TXT File"

        # ✅ PDF
        elif pdf_file and pdf_file.filename != "":
            reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += (page.extract_text() or "") + "\n"
            extracted_text = pdf_text
            source_used = "PDF File"

        # ✅ DOCX
        elif docx_file and docx_file.filename != "":
            doc = Document(docx_file)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])

        # ✅ IMAGE (OCR)
        elif image_file and image_file.filename != "":
            img = Image.open(image_file).convert("RGB")
            extracted_text = pytesseract.image_to_string(img, lang="eng")
            source_used = "Image (OCR)"


        # ✅ AUDIO (Whisper)
        elif audio_file and audio_file.filename != "":
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio:
                audio_file.save(temp_audio.name)
                result = whisper_model.transcribe(temp_audio.name)
                extracted_text = result["text"]
                source_used = "Audio (Whisper)"

        # ✅ VIDEO (extract audio → Whisper)
        elif video_file and video_file.filename != "":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                video_file.save(temp_video.name)

                clip = VideoFileClip(temp_video.name)
                clip.close()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    clip.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)

                    result = whisper_model.transcribe(temp_audio.name)
                    extracted_text = result["text"]
                    source_used = "Video (Whisper)"

        # ✅ LIMIT extracted text (prevents 413 error)
        extracted_text = extracted_text.strip()[:6000]

        # ✅ Build prompt
        if extracted_text:
            prompt = f"""
        You are an expert exam question setter.

        Task:
        Generate exactly {count} high-quality MCQs from the given text.

        Difficulty: {difficulty}

        Rules:
        1. Output format MUST be:
        Q1) ...
        A) ...
        B) ...
        C) ...
        D) ...
        Answer: <A/B/C/D>
        Explanation: <1 line>

        2. Questions must be from the text only.
        3. Do not repeat questions.
        4. Do not add extra commentary.

        TEXT:
        {extracted_text}
        """
        else:
            prompt = f"""
        You are an expert exam question setter.

        Generate exactly {count} MCQs on this topic: {topic}
        Difficulty: {difficulty}

        Format:
        Q1) ...
        A) ...
        B) ...
        C) ...
        D) ...
        Answer: <A/B/C/D>
        Explanation: <1 line>
        """


        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        mcqs = response.choices[0].message.content
        mcqs = f"Source Used: {source_used}\n\n" + mcqs

    return render_template("index.html", mcqs=mcqs)


if __name__ == "__main__":
    app.run(debug=True)
