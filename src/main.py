"""
main.py
-------
The single Streamlit app that brings together the base chatbot and all 6
internship tasks. Run it with:  streamlit run main.py  (from inside the src folder)

The sidebar lets you choose which feature to use. Each feature maps to one of
the task files in the tasks/ folder.
"""

import streamlit as st
from PIL import Image

# Base project pieces.
from langchain_helper import create_vector_db, get_qa_chain

# The six task modules.
from tasks import task1_knowledge_base as t1
from tasks import task2_multimodal as t2
from tasks import task3_medical_qa as t3
from tasks import task4_arxiv_expert as t4
from tasks import task5_sentiment as t5
from tasks import task6_multilanguage as t6

from langchain_helper import llm

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("Customer Service Chatbot 🤖")

# Sidebar menu to switch between features.
feature = st.sidebar.radio(
    "Choose a feature",
    [
        "Base Q&A Chatbot",
        "Task 1: Expand Knowledge Base",
        "Task 2: Multi-modal (Image) Chat",
        "Task 3: Medical Q&A",
        "Task 4: arXiv Expert",
        "Task 5: Sentiment-aware Reply",
        "Task 6: Multi-language Reply",
    ],
)

# ---------------------------------------------------------------- Base Q&A
if feature == "Base Q&A Chatbot":
    st.header("Ask about our courses / services")
    if st.button("Create / Rebuild Knowledge Base"):
        with st.spinner("Building knowledge base..."):
            create_vector_db()
        st.success("Knowledge base ready.")

    question = st.text_input("Your question:")
    if question:
        chain = get_qa_chain()
        response = chain(question)
        st.subheader("Answer")
        st.write(response["result"])

# ---------------------------------------------------------------- Task 1
elif feature == "Task 1: Expand Knowledge Base":
    st.header("Task 1: Add new information to the knowledge base")
    st.write("Add a web page or a CSV file. New info is merged into the database.")

    url = st.text_input("Web page URL to add:")
    if st.button("Add this URL") and url:
        with st.spinner("Fetching and adding..."):
            n = t1.update_from_url(url)
        st.success(f"Added {n} chunks from the page.")

    csv_path = st.text_input("CSV file path to add (columns: prompt, response):")
    if st.button("Add this CSV") and csv_path:
        with st.spinner("Reading and adding..."):
            n = t1.update_from_csv(csv_path)
        st.success(f"Added {n} rows from the CSV.")

# ---------------------------------------------------------------- Task 2
elif feature == "Task 2: Multi-modal (Image) Chat":
    st.header("Task 2: Ask about an image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    img_question = st.text_input("Question about the image (optional):")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Your image", use_column_width=True)
        if st.button("Ask Gemini about this image"):
            with st.spinner("Looking at the image..."):
                answer = t2.describe_image(image, img_question)
            st.subheader("Answer")
            st.write(answer)

# ---------------------------------------------------------------- Task 3
elif feature == "Task 3: Medical Q&A":
    st.header("Task 3: Medical Q&A (MedQuAD)")
    st.info("First build the medical database (see README) before asking.")
    med_q = st.text_input("Medical question:")
    if med_q:
        try:
            result = t3.answer_medical_question(med_q)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Recognised medical terms")
            st.json(result["entities"])
        except FileNotFoundError as e:
            st.error(str(e))

# ---------------------------------------------------------------- Task 4
elif feature == "Task 4: arXiv Expert":
    st.header("Task 4: arXiv research expert")
    st.info("First build the arXiv database (see README) before searching.")

    search_q = st.text_input("Search papers about:")
    if search_q:
        try:
            papers = t4.search_papers(search_q)
            for p in papers:
                st.markdown(f"**{p['title']}**")
                st.caption(p["snippet"])
        except FileNotFoundError as e:
            st.error(str(e))

    concept = st.text_input("Explain a concept:")
    if concept:
        st.write(t4.explain_concept(concept))

# ---------------------------------------------------------------- Task 5
elif feature == "Task 5: Sentiment-aware Reply":
    st.header("Task 5: Reply changes with the user's mood")
    msg = st.text_input("Type a customer message:")
    if msg:
        sentiment = t5.detect_sentiment(msg)
        st.write(f"Detected sentiment: **{sentiment}**")
        prompt = (
            f"{t5.tone_instruction(sentiment)}\n\nCustomer message: {msg}\n\n"
            "Write a helpful reply."
        )
        st.subheader("Reply")
        st.write(llm.invoke(prompt))

# ---------------------------------------------------------------- Task 6
elif feature == "Task 6: Multi-language Reply":
    st.header("Task 6: Reply in the user's language")
    msg = st.text_input("Type a message in any language:")
    if msg:
        language = t6.detect_language(msg)
        st.write(f"Detected language: **{language}**")
        prompt = (
            f"{t6.language_instruction(language)}\n\nUser message: {msg}\n\n"
            "Write a helpful reply."
        )
        st.subheader("Reply")
        st.write(llm.invoke(prompt))
