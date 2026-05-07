import streamlit as st

def render_sidebar() -> dict:
    """
    Renders the sidebar for patient data entry with a polished medical form look.
    """
    with st.sidebar:
        st.header("🏥 Patient Data")
        st.info("Enter demographics for the official report.")
        
        st.markdown("### Demographics")
        # Added placeholder for better UX
        name = st.text_input("Full Name", key="name", placeholder="e.g. John Doe")
        
        # Use columns to save vertical space
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=60, key="age")
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
            
        st.markdown("### Clinical Context")
        # Added optional history field (looks professional)
        history = st.text_area(
            "Medical History (Optional)", 
            height=100, 
            placeholder="e.g. Previous knee surgery in 2019, chronic pain..."
        )
        
        st.markdown("---")
        st.caption("🔒 Data processed locally on device.")
        
    return {
        "name": name,
        "age": str(age),
        "gender": gender,
        "history": history
    }

def render_upload():
    """
    Renders the file uploader with clear instructions.
    """
    st.subheader("1. Image Acquisition")
    
    # Styled container for the uploader
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload Anteroposterior (AP) Knee X-ray",
            type=["jpg", "jpeg", "png", "dcm"],
            help="Supported formats: DICOM (Medical Standard), PNG, JPEG",
            key="uploader"
        )
        
    if not uploaded_file:
        st.info("👆 Please upload an X-ray image to initialize the diagnostic pipeline.")
        
    return uploaded_file