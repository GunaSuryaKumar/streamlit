import streamlit as st
import pandas as pd
import os
import uuid
import pdfplumber
from mistralai import Mistral

def extract_tables_from_pdf(filename):
    """
    Extracts tables from a PDF using pdfplumber and ensures correct column formatting.

    Args:
        filename (str): Path to the PDF file.

    Returns:
        List[pd.DataFrame]: A list of cleaned DataFrames for each extracted table.
    """
    dfs = []
    
    st.write(f"📄 Opening PDF: {filename}")

    with pdfplumber.open(filename) as pdf:
        total_pages = len(pdf.pages)
        st.write(f"📖 PDF Loaded Successfully! Total Pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):
            st.write(f"🔍 Extracting tables from Page {page_num}...")
            tables = page.extract_tables()

            if not tables:
                st.write(f"❌ No tables found on Page {page_num}")
                continue

            for table_idx, table in enumerate(tables, start=1):
                df = pd.DataFrame(table)

                # Drop completely empty rows
                df.dropna(how="all", inplace=True)

                # If the first row has more than 50% NaNs, discard it as header
                if df.iloc[0].isna().sum() > len(df.columns) // 2:
                    df = df[1:].reset_index(drop=True)

                # Ensure all column names are unique
                df.columns = [f"Column_{i}" if not isinstance(col, str) or col.strip() == "" else col.strip()
                              for i, col in enumerate(df.iloc[0])]

                # Drop first row if it was mistakenly used as column names
                df = df[1:].reset_index(drop=True)

                dfs.append(df)
                st.write(f"✅ Extracted Table {table_idx} from Page {page_num} with {len(df.columns)} columns")

    st.write(f"🎯 Extraction Complete! Total Tables Extracted: {len(dfs)}")
    return dfs

def summarize_table(table_text, max_new_tokens=100):
    """
    Summarizes table data using an LLM (Mistral API).

    Args:
        table_text (str): Text extracted from the table.

    Returns:
        str: A concise summary.
    """
    prompt = (
        "Summarize the following table data:\n\n" 
        f"{table_text}\n\n"
        "Provide a short, structured summary."
    )

    api_key = "MMNlnPxuMxBfeIGG4pIGIfSwBdIgjlVA"  # Secure your API Key

    client = Mistral(api_key=api_key)

    st.write(f"⚡ Generating summary for extracted table...")
    
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write(f"📝 Summary Generated Successfully!")
    return chat_response.choices[0].message.content

def main():
    st.set_page_config(
        page_title="📄 PDF Table Extraction & Summarization",
        layout="wide",
        page_icon="📈",
    )

    st.markdown("""
        <style>
            .header-container {
                background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .header {
                font-size: 2.5em;
                font-weight: 800;
                text-align: center;
                background: linear-gradient(120deg, #0D6EFD 0%, #0B5ED7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-container">
            <h1 class="header">📄 PDF Table Extraction and Summarization</h1>
        </div>
    """, unsafe_allow_html=True)

    if 'temp_filename' not in st.session_state:
        st.session_state.temp_filename = None

    with st.sidebar:
        st.markdown('<h2 style="color: white;">Upload Your PDF</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Delete previous temporary file if it exists
            if st.session_state.temp_filename and os.path.exists(st.session_state.temp_filename):
                try:
                    os.remove(st.session_state.temp_filename)
                except Exception as e:
                    st.warning(f"Could not delete previous temporary file: {e}")

            st.session_state.temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
            with open(st.session_state.temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ PDF Uploaded Successfully!")
            st.write(f"📂 PDF Uploaded: {uploaded_file.name}")

    if uploaded_file is not None:
        try:
            st.markdown("## 📊 Extracted Tables")
            dfs = extract_tables_from_pdf(st.session_state.temp_filename)

            if not dfs:
                st.warning("No tables found in the PDF.")
            else:
                for idx, df in enumerate(dfs):
                    st.markdown(f"### Table {idx+1}")
                    st.dataframe(df)

                    table_text = df.to_string()
                    summary = summarize_table(table_text)
                    st.markdown(f"**Summary:** {summary}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
