import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import json

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key="AIzaSyD-pDdkdydHeVfsnNy9YAmQoIthM7--iMM", model="gemini-pro")
embeddings_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyD-pDdkdydHeVfsnNy9YAmQoIthM7--iMM", model="models/embedding-001")

# Load PDF and create FAISS vectorstore
pdf = PyPDFLoader(file_path="C:\\Users\\lokes\\Desktop\\cyberaware\\rag_test.pdf")
documents = pdf.load()
vectorstore_db = FAISS.from_documents(documents=documents, embedding=embeddings_model)

# Streamlit UI
st.title("Cyber Security Awareness Platform")
question = st.text_input(label="What is your question?")
submit = st.button(label="Generate Quiz")

# Ensure session state is initialized
if "questions" not in st.session_state:
    st.session_state.questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if submit:
    documents_list = vectorstore_db.similarity_search(question)
    text = "\n".join([doc.page_content for doc in documents_list])

    # AI Prompt to enforce JSON structure
    prompt = f"""
    Context Documents: {text}
    Generate 10 quiz questions related to the given question: "{question}".
    Return the result **ONLY** in JSON format like this:
    [
        {{
            "question": "What is the capital of France?",
            "options": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": "Paris"
        }},
        {{
            "question": "Who developed Python?",
            "options": ["Guido van Rossum", "Elon Musk", "Mark Zuckerberg", "Dennis Ritchie"],
            "answer": "Guido van Rossum"
        }}
    ]
    NO additional text, just JSON.
    """

    response = llm.invoke(prompt)

    try:
        parsed_response = json.loads(response.content.strip())

        # Ensure it's a list
        if isinstance(parsed_response, list):
            st.session_state.questions = parsed_response
        else:
            st.session_state.questions = []
            st.error("Invalid AI response format. Please try again.")
    except json.JSONDecodeError:
        st.error("Failed to parse AI response as JSON. Please try again.")
        st.session_state.questions = []

# Display questions and collect answers
if st.session_state.questions:
    st.subheader("Quiz Questions")
    
    for i, q in enumerate(st.session_state.questions):
        question_text = q.get("question", f"Question {i+1}")  # Fallback if missing
        
        # Ensure options exist
        options = q.get("options", [])
        if not options or len(options) < 2:
            options = ["Option A", "Option B", "Option C", "Option D"]  # Default fallback options

        # Display the question with radio button options
        st.session_state.user_answers[i] = st.radio(question_text, options, key=f"q{i}")

    # Add the Submit Quiz button **AFTER** questions are displayed
    if st.button("Submit Quiz"):
        correct_count = 0
        total_questions = len(st.session_state.questions)

        # Display results
        st.subheader("Quiz Results:")
        for i, q in enumerate(st.session_state.questions):
            correct_answer = q.get("answer", "Unknown")
            user_answer = st.session_state.user_answers.get(i, "No Answer")

            # Check if correct
            if user_answer == correct_answer:
                correct_count += 1
                st.write(f"✅ **{q['question']}**")
                st.write(f"Your Answer: **{user_answer}**")
            else:
                st.write(f"❌ **{q['question']}**")
                st.write(f"Your Answer: **{user_answer}**")
                st.write(f"Correct Answer: **{correct_answer}**")

            st.write("---")  # Separator for clarity

        # Display total score
        st.success(f"🎯 You scored {correct_count} / {total_questions}!")
