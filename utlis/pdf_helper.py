from fpdf import FPDF
from datetime import datetime
from config.settings import KL_GRADES
import os

class ReportPDF(FPDF):
    def header(self):
        # Professional Header
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Osteo-Analyst: AI Radiological Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        # Footer with page number
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"AI-generated report. For clinical review only. Page {self.page_no()}", 0, 0, "C")

def generate_report(
    patient_name: str,
    age: str,
    gender: str,
    kl_grade: int,
    confidence: float,
    findings_text: str,
    cam_image_path: str,
    output_path: str
):
    """
    Generates a PDF report combining patient data, AI prediction, 
    Grad-CAM visualization, and LLM text.
    """
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Patient Demographics Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Information", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Name: {patient_name}", 0, 1)
    pdf.cell(0, 6, f"Age: {age}  |  Gender: {gender}", 0, 1)
    pdf.ln(5)

    # 2. AI Diagnosis Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Diagnostic Summary", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Predicted KL Grade: {kl_grade} - {KL_GRADES.get(kl_grade, 'Unknown')}", 0, 1)
    pdf.cell(0, 6, f"Model Confidence: {confidence:.1%}", 0, 1)
    pdf.ln(5)

    # 3. Visualization Section (Grad-CAM)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Radiological Evidence (AI Attention Map)", 0, 1)
    # Check if image exists before adding
    if os.path.exists(cam_image_path):
        # x=15 puts it slightly indented, w=180 fits A4 width
        pdf.image(cam_image_path, x=15, w=180)
        pdf.ln(5) # Space after image (height is auto-handled by FPDF usually, but specific logic helps)
    else:
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "[Error: Grad-CAM Image not found]", 0, 1)
    
    # Add a large break after image to ensure text doesn't overlap
    # Note: 180mm width image typically takes ~150-180mm height depending on aspect ratio.
    # We can rely on FPDF's flow or force a break.
    pdf.ln(10)

    # 4. Findings Section (LLM Text)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Findings", 0, 1)
    pdf.set_font("Arial", size=11)
    # multi_cell handles wrapping
    pdf.multi_cell(0, 6, findings_text)
    pdf.ln(10)

    # 5. Timestamp
    pdf.set_font("Arial", "I", 9)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f"Report Generated on: {timestamp}", 0, 1)

    # Save
    pdf.output(output_path)