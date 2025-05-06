# Resume & Job Description Matcher

A sophisticated application that leverages Natural Language Processing (NLP) and Machine Learning techniques to analyze and compare resumes against job descriptions. This tool provides valuable insights for both recruiters and job seekers by quantifying the match between candidate qualifications and job requirements.

Drive Link - https://drive.google.com/drive/folders/1nstW7VP-R9l6Typ2Vy0rFCKSyPhIWkEg?usp=sharing
## Core Functionality

### 1. Multi-Resume Analysis
- **Input**: Multiple resumes in PDF or DOCX format
- **Processing**: Parallel analysis of multiple candidate profiles
- **Output**: Ranked list of candidates based on job description match

### 2. Intelligent Text Processing
- **Document Parsing**: Extracts text from PDF and DOCX formats
- **Text Preprocessing**:
  - Tokenization: Breaks text into meaningful units
  - Stopword Removal: Eliminates common, non-meaningful words
  - Lemmatization: Reduces words to their base form
  - Case Normalization: Standardizes text case

### 3. Matching Algorithm
The application uses a sophisticated matching algorithm that combines multiple NLP techniques:

#### TF-IDF Vectorization
- **Term Frequency (TF)**: Measures how often terms appear in documents
- **Inverse Document Frequency (IDF)**: Weights terms based on their uniqueness
- **Vector Space Model**: Converts text into numerical vectors for comparison

#### Cosine Similarity
- **Calculation**: Measures the cosine of the angle between document vectors
- **Range**: 0 to 1 (0% to 100% match)
- **Interpretation**: Higher scores indicate better matches

### 4. Key Term Analysis
- **Extraction**: Identifies the most significant terms in both documents
- **Scoring**: Calculates term importance using TF-IDF
- **Comparison**: Highlights matching and missing key terms

## Technical Implementation

### Data Processing Pipeline
1. **Document Input**
   - PDF parsing using pdfplumber
   - DOCX parsing using python-docx
   - Text extraction and cleaning

2. **Text Analysis**
   - NLTK for text preprocessing
   - scikit-learn for vectorization
   - Custom algorithms for term importance

3. **Similarity Calculation**
   - TF-IDF vectorization
   - Cosine similarity computation
   - Score normalization and ranking

### Key Metrics

#### 1. Match Score
- **Calculation**: Cosine similarity between resume and job description vectors
- **Range**: 0-100%
- **Interpretation**: 
  - 80-100%: Excellent match
  - 60-79%: Good match
  - 40-59%: Moderate match
  - 0-39%: Poor match

#### 2. Key Terms
- **Extraction**: Top 10 most significant terms from each document
- **Scoring**: TF-IDF importance score
- **Analysis**: Term overlap and gap analysis

#### 3. Comparative Analysis
- **Score Differences**: Relative performance between candidates
- **Term Coverage**: Missing and matching key terms
- **Recommendations**: Specific improvements for each resume

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume-job-matcher
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the application:
```bash
streamlit run app.py
```

2. Upload resumes and enter job description
3. Click "Analyze Resumes" to begin processing
4. Review the comprehensive analysis:
   - Overall rankings
   - Detailed term analysis
   - Improvement recommendations
   - Comparative insights

## Technical Requirements

- Python 3.8+
- Streamlit 1.32.0
- NLTK 3.8.1
- scikit-learn 1.4.0
- pdfplumber 0.10.3
- python-docx 1.1.0
- pandas 2.2.0

## Best Practices

1. **Resume Formatting**
   - Use clear, structured formats
   - Include relevant keywords
   - Maintain consistent formatting

2. **Job Description**
   - Be specific about requirements
   - Include key skills and qualifications
   - Use industry-standard terminology

3. **Analysis Interpretation**
   - Consider context beyond scores
   - Review key term matches
   - Use recommendations for improvement

## Contributing

We welcome contributions to improve the application. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
