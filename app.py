import streamlit as st
import pdfplumber
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

def calculate_similarity(resume_text, job_desc_text):
    # Preprocess both texts
    resume_processed = preprocess_text(resume_text)
    job_desc_processed = preprocess_text(job_desc_text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_desc_processed])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity * 100  # Convert to percentage

def extract_key_terms(text, top_n=10):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores
    scores = tfidf_matrix.toarray()[0]
    
    # Create a dictionary of terms and their scores
    term_scores = dict(zip(feature_names, scores))
    
    # Sort by score
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_terms

# Set page config
st.set_page_config(
    page_title="Resume & Job Description Matcher",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        color: #34495e;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    .stMarkdown {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Resume & Job Description Matcher")

st.markdown("""
### Overview
This application provides an intelligent analysis of resume-job description compatibility through advanced text processing and matching algorithms.
""")

# Create two columns for features and instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Key Features
    - **Multi-Resume Analysis**: Upload and compare multiple resumes against a job description
    - **Automated Scoring**: Advanced algorithm calculates compatibility scores
    - **Key Term Analysis**: Identifies critical skills and qualifications
    - **Comparative Analysis**: Detailed comparison between multiple candidates
    """)

with col2:
    st.markdown("""
    ### How to Use
    1. Upload one or more resumes (PDF or DOCX format)
    2. Enter the job description
    3. Click 'Analyze Resumes' to begin processing
    4. Review the results and detailed analysis
    """)

# File upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resumes")
    resume_files = st.file_uploader("Select resume files (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

with col2:
    st.subheader("Job Description")
    job_desc = st.text_area("Enter the job description", height=300)

# Add analyze button
analyze_button = st.button("Analyze Resumes", type="primary")

# Process files and calculate matching scores
if analyze_button and resume_files and job_desc:
    try:
        with st.spinner("Analyzing resumes..."):
            # Store resume data
            resume_data = []
            
            # Process each resume
            for resume_file in resume_files:
                # Extract text from resume
                if resume_file.name.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(resume_file)
                else:
                    resume_text = extract_text_from_docx(resume_file)
                
                # Calculate similarity score
                similarity_score = calculate_similarity(resume_text, job_desc)
                
                # Extract key terms
                resume_terms = extract_key_terms(resume_text)
                
                # Store resume data
                resume_data.append({
                    'name': resume_file.name,
                    'text': resume_text,
                    'score': similarity_score,
                    'terms': resume_terms
                })
            
            # Sort resumes by score
            resume_data.sort(key=lambda x: x['score'], reverse=True)
            
            # Store the processed data in session state
            st.session_state.resume_data = resume_data
            st.session_state.job_desc = job_desc
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        # Clear any existing data in case of error
        if 'resume_data' in st.session_state:
            del st.session_state.resume_data
        if 'job_desc' in st.session_state:
            del st.session_state.job_desc

# Display results if we have data in session state
if 'resume_data' in st.session_state and st.session_state.resume_data:
    resume_data = st.session_state.resume_data
    job_desc = st.session_state.job_desc
    
    # Display overall results
    st.markdown("---")
    st.subheader("Resume Rankings")
    
    # Create a DataFrame for the rankings
    rankings_df = pd.DataFrame([
        {'Resume': data['name'], 'Match Score': f"{data['score']:.1f}%"}
        for data in resume_data
    ])
    st.dataframe(rankings_df, hide_index=True)
    
    # Allow user to select a resume for detailed analysis
    st.markdown("---")
    st.subheader("Detailed Analysis")
    
    # Create a list of resume names for selection
    resume_names = [data['name'] for data in resume_data]
    
    # Use session state to maintain selection
    if 'selected_resume' not in st.session_state:
        st.session_state.selected_resume = resume_names[0] if resume_names else None
    
    # Update the selectbox with the current selection
    selected_resume = st.selectbox(
        "Select a resume for detailed analysis",
        options=resume_names,
        index=resume_names.index(st.session_state.selected_resume) if st.session_state.selected_resume in resume_names else 0
    )
    
    # Update session state with new selection
    st.session_state.selected_resume = selected_resume
    
    # Get the selected resume data
    selected_data = next(data for data in resume_data if data['name'] == selected_resume)
    
    # Display detailed analysis for selected resume
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Terms in Selected Resume")
        resume_df = pd.DataFrame(selected_data['terms'], columns=['Term', 'Importance'])
        st.dataframe(resume_df, hide_index=True)
    
    with col2:
        st.subheader("Key Terms in Job Description")
        job_desc_terms = extract_key_terms(job_desc)
        job_df = pd.DataFrame(job_desc_terms, columns=['Term', 'Importance'])
        st.dataframe(job_df, hide_index=True)
    
    # Recommendations for selected resume
    st.subheader("Recommendations")
    missing_terms = [term for term, _ in job_desc_terms if term not in [r[0] for r in selected_data['terms']]]
    if missing_terms:
        st.write("Consider adding these important terms to your resume:")
        st.write(", ".join(missing_terms))
    else:
        st.write("Great job! Your resume covers all the key terms from the job description.")
    
    # Display comparison with other resumes
    st.markdown("---")
    st.subheader("Comparison with Other Resumes")
    
    # Create comparison DataFrame
    comparison_data = []
    for data in resume_data:
        if data['name'] != selected_resume:
            comparison_data.append({
                'Resume': data['name'],
                'Score Difference': f"{data['score'] - selected_data['score']:+.1f}%"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)

elif analyze_button and (not resume_files or not job_desc):
    st.warning("Please upload at least one resume and provide a job description before analysis.")
else:
    st.info("Upload resumes and enter a job description to begin analysis.") 