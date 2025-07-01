import streamlit as st
from openai import OpenAI
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import spacy
from fpdf import FPDF

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Extract text from uploaded PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Extract NER-based keywords
def extract_keywords(text):
    doc = nlp(text)
    return list(
        set(
            [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "SKILL"]
            ]
        )
    )


# Compare keywords between JD and resume
def compare_keywords(jd_text, resume_text):
    jd_keywords = set(extract_keywords(jd_text))
    resume_keywords = set(extract_keywords(resume_text))
    missing_keywords = list(jd_keywords - resume_keywords)
    match_score = (
        int((len(jd_keywords & resume_keywords) / len(jd_keywords)) * 100)
        if jd_keywords
        else 0
    )
    return match_score, missing_keywords


# Custom ATS Score Calculation
def calculate_custom_ats_score(resume_text, job_description, parsed_json):
    score = 0
    total = 5

    match_keywords = [kw.lower() for kw in parsed_json.get("MissingKeywords", [])]
    score += 1 if len(match_keywords) <= 3 else 0
    score += 1 if len(resume_text.split()) >= 250 else 0

    sections = ["Education", "Experience", "Skills", "Projects", "Certifications"]
    section_score = sum(
        1 for sec in sections if sec.lower() in resume_text.lower()
    ) / len(sections)
    score += section_score

    score += 1 if 300 <= len(resume_text.split()) <= 800 else 0
    score += 1 if int(parsed_json["JD Match"].replace("%", "")) >= 80 else 0

    return int((score / total) * 100)


# OpenAI Prompt Builder
def build_prompt(resume_text, job_description):
    return f"""
You are a professional ATS system and resume reviewer.

Your tasks are:
1. Analyze how well the resume matches the provided job description (JD Match %).
2. Assign an ATS Score based on structure, formatting, keyword usage, and scannability.
3. Extract and analyze key sections from the resume: 'Education', 'Experience', 'Skills', 'Projects', and 'Certifications'.
4. For each section, suggest improvements tailored to the given job description. If a section is missing or weak, clearly state that.
5. Identify missing keywords from the job description.
6. Generate a customized profile summary based on the resume and job description.

Format your response strictly in the following JSON structure:
{{
  "JD Match": "85%",
  "ATS Score": "78%",
  "MissingKeywords": ["DevOps", "Cloud", "Linux"],
  "Profile Summary": "Customized summary here...",
  "ImprovementSuggestions": {{
    "Education": "Suggestions for the Education section",
    "Experience": "Suggestions for the Experience section",
    "Skills": "Suggestions for the Skills section",
    "Projects": "Suggestions for the Projects section",
    "Certifications": "Suggestions for the Certifications section"
  }}
}}

Resume:
{resume_text}

Job Description:
{job_description}
"""


# Get OpenAI response
def get_openai_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    return response.choices[0].message.content


# Interpret JD Match
def evaluate_jd_match(match_percent, missing_keywords):
    if match_percent >= 85 and len(missing_keywords) <= 2:
        return "✅ Your resume is well-matched to the job description.", "Good"
    elif match_percent >= 70:
        return (
            "⚠️ Your resume matches moderately. You can improve it further.",
            "Moderate",
        )
    else:
        return (
            "❌ Your resume needs major improvement to match the job description.",
            "Poor",
        )


# Interpret ATS Score
def evaluate_ats_score(score):
    if score >= 85:
        return "✅ ATS-ready and well-formatted."
    elif score >= 70:
        return "⚠️ Resume is decent, but could use formatting or clarity improvements."
    else:
        return "❌ Resume structure or clarity needs major improvement."


# Build updated resume with suggestions and keywords
def build_updated_resume(parsed, original_text):
    updated_resume = ""

    updated_resume += "## Profile Summary\n" + parsed["Profile Summary"] + "\n\n"
    suggestions = parsed.get("ImprovementSuggestions", {})

    for section in ["Education", "Experience", "Skills", "Projects", "Certifications"]:
        updated_resume += f"## {section}\n"
        updated_resume += suggestions.get(section, "") + "\n\n"

    if parsed.get("MissingKeywords"):
        updated_resume += "## Additional Relevant Skills\n"
        updated_resume += ", ".join(parsed["MissingKeywords"]).title()

    return updated_resume


# Save PDF from text
def save_pdf_from_text(text, filename="Updated_Resume.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, txt=line)
    os.makedirs("temp", exist_ok=True)
    pdf_path = os.path.join("temp", filename)
    pdf.output(pdf_path)
    return pdf_path


# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="Smart ATS Analyzer", layout="centered")
st.title("📄 Smart ATS Analyzer")
st.markdown(
    "Upload your resume and job description to receive AI-based feedback and an improved downloadable version."
)

jd = st.text_area("🧾 Paste the Job Description")
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type="pdf")

submit = st.button("🚀 Analyze Resume")

if submit:
    if uploaded_file is None or jd.strip() == "":
        st.warning("⚠️ Please upload a resume and paste the job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = input_pdf_text(uploaded_file)
            full_prompt = build_prompt(resume_text, jd)
            response = get_openai_response(full_prompt)

        st.subheader("🔍 ATS Feedback")

        try:
            parsed = json.loads(response)

            match_percent = int(parsed["JD Match"].replace("%", ""))
            ats_score_llm = int(parsed["ATS Score"].replace("%", ""))
            missing_keywords = parsed["MissingKeywords"]

            st.markdown(f"### 🎯 JD Match (GPT): `{parsed['JD Match']}`")
            st.progress(match_percent)
            st.markdown(f"**{evaluate_jd_match(match_percent, missing_keywords)[0]}**")

            st.markdown(f"### 🤖 ATS Score (GPT): `{parsed['ATS Score']}`")
            st.progress(ats_score_llm)
            st.markdown(f"**{evaluate_ats_score(ats_score_llm)}**")

            custom_score = calculate_custom_ats_score(resume_text, jd, parsed)
            st.markdown(f"### 🧠 ATS Score (Enhanced): `{custom_score}%`")
            st.progress(custom_score)
            st.markdown(evaluate_ats_score(custom_score))

            match_ner, missing_ner = compare_keywords(jd, resume_text)
            st.markdown(f"### 🔎 JD Match (NER-based): `{match_ner}%`")
            st.progress(match_ner)

            st.markdown("### ❌ Missing Keywords (GPT):")
            if missing_keywords:
                for kw in missing_keywords:
                    st.markdown(f"- {kw}")
            else:
                st.markdown("_No missing keywords — great match!_")

            st.markdown("### 📝 Profile Summary:")
            st.markdown(parsed["Profile Summary"])

            st.markdown("### 💡 Section-wise Suggestions:")
            for section, suggestion in parsed.get("ImprovementSuggestions", {}).items():
                with st.expander(f"📌 {section}"):
                    st.markdown(suggestion)

            st.markdown("### 📄 Updated Resume Preview:")
            updated_text = build_updated_resume(parsed, resume_text)
            st.text_area("Updated Resume", updated_text, height=400)

            pdf_path = save_pdf_from_text(updated_text)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Updated Resume",
                    f,
                    file_name="Updated_Resume.pdf",
                    mime="application/pdf",
                )

        except Exception as e:
            st.error("⚠️ Failed to parse OpenAI response. Raw output below:")
            st.code(response)
            st.exception(e)
