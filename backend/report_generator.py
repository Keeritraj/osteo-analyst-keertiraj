4import os

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from utils.pdf_helper import generate_report
from config.settings import SYSTEM_PROMPT, KL_GRADES

def generate_llm_report(patient_data: dict, kl_grade: int, confidence: float) -> str:
    try:
       
        llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", (
                f"Patient Name: {patient_data['name']}. "
                f"Age: {patient_data['age']}. Gender: {patient_data['gender']}.\n"
                f"Model Prediction: Grade {kl_grade} ({KL_GRADES[kl_grade]}).\n"
                f"Confidence: {confidence:.1%}."
            ))
        ])
        
        chain = prompt | llm
        
       
        response = chain.invoke({})
        
     
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
            
    except Exception as e:
       
        print(f"LLM Error: {e}")
        fallback = (
            f"Findings: AI model predicted KL Grade {kl_grade} ({KL_GRADES[kl_grade]}) "
            f"with {confidence:.1%} confidence. Grad-CAM heatmap indicates regions of interest.\n"
            f"Impression: Preliminary AI assessment. Requires radiologist review."
        )
        return fallback

def create_final_report(
    patient_data: dict,
    kl_grade: int,
    confidence: float,
    cam_image_path: str,
    output_pdf_path: str
):
    findings = generate_llm_report(patient_data, kl_grade, confidence)
    generate_report(
        patient_name=patient_data["name"],
        age=patient_data["age"],
        gender=patient_data["gender"],
        kl_grade=kl_grade,
        confidence=confidence,
        findings_text=findings,
        cam_image_path=cam_image_path,
        output_path=output_pdf_path
    )