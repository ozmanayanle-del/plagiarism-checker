import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Document Plagiarism Checker")

st.write("Compare two documents using text similarity.")

document1 = st.text_area("Enter the first document")
document2 = st.text_area("Enter the second document")

if st.button("Check Similarity"):
    if not document1.strip() or not document2.strip():
        st.warning("Please enter both documents.")
    else:
        documents = [document1, document2]

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1], matrix[1:2]
        )[0][0]

        percentage = similarity * 100

        st.subheader("Result")
        st.metric("Text Similarity", f"{percentage:.2f}%")

        if percentage >= 70:
            st.warning("High similarity detected.")
        elif percentage >= 40:
            st.info("Moderate similarity detected.")
        else:
            st.success("Low similarity detected.")

        st.caption(
            "Similarity alone does not prove plagiarism."
        )