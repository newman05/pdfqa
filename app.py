# Streamlit is a Python library that makes it easy to create interactive web apps
# without needing JavaScript, HTML, or CSS knowledge. It turns Python scripts into shareable web apps.
import streamlit as st 
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



from htmlTemplates import css, bot_template, user_template

load_dotenv() 
def get_pdf_text(pdf_docs):
    text = "" 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 1000,

        chunk_overlap=200, # context ke liye, nahi hone aadha text koi aur chunk me hoga aur aadha kahi aur
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest",
            temperature=0.3
        )
        # quick sanity check
        llm.invoke("ping")
        print("[INFO] Using Gemini Flash")
        return llm
    except Exception as e:
        print(f"[WARN] Gemini failed, falling back to Mistral: {e}")
        return ChatOllama(
            model="mistral",
            temperature=0.3
        )


def get_conversation_chain(vectorstore):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
       ("system", "Answer the question using ONLY the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
    )

    # store chat history per session
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversation_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversation_chain

def handle_userinput(user_question):
    try:
        print(f"[DEBUG] User asked: {user_question}")
        
        # store user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        print("[DEBUG] Invoking conversation chain...")
        response = st.session_state.conversation.invoke(
            {"question": user_question},
            config={"configurable": {"session_id": "default"}}
        )
        
        print(f"[DEBUG] Response received: {response}")
        print(f"[DEBUG] Response type: {type(response)}")

        # Handle both string and AIMessage responses
        bot_response = response.content if hasattr(response, 'content') else str(response)
        print(f"[DEBUG] Bot response text: {bot_response}")
        
        st.session_state.messages.append(
            {"role": "bot", "content": bot_response}
        )
        print("[DEBUG] Message stored in session state")
        
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        print(f"[ERROR] Full error: {e}")
        import traceback
        traceback.print_exc()


def main() : 
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # st.set_page_config() configures the page settings like title shown in browser tab and favicon icon
    st.set_page_config(page_title="pdfchat", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # st.header() displays a large heading/title on the web page
    st.header("pdfchat :books:")

    # st.text_input() creates an input field where users can type text (in this case, to ask a question)
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("ask question about your docs", key="user_input_field")
    with col2:
        submit_button = st.button("Send")

    if submit_button and user_question and st.session_state.conversation:
        with st.spinner("Waiting for response... (this may take 30-60 seconds)"):
            handle_userinput(user_question)

            st.rerun()  # Force Streamlit to rerun and display the message


   # st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
   # st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)
    # render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.write(
            user_template.replace(
            "{{MSG}}",
            "\n".join(msg["content"]) if isinstance(msg["content"], list) else str(msg["content"])
            ),

            unsafe_allow_html=True
        )
        else:
            st.write(
            bot_template.replace(
            "{{MSG}}",
            "\n".join(msg["content"]) if isinstance(msg["content"], list) else str(msg["content"])
            ),

            unsafe_allow_html=True
        )


    with st.sidebar: 
        # st.subheader() displays a subheader in the sidebar of the web app
        st.subheader("Your Documents")
        # st.file_uploader() creates a file upload widget in the sidebar for users to upload PDF files
        pdf_docs = st.file_uploader("Upload your pdf files", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                try:
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from PDFs. Make sure they contain text.")
                        return

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text) 
                    st.info(f"Created {len(text_chunks)} text chunks")

                    # create vector store
                    st.info("Creating embeddings... (this may take a moment)")
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Vector store created ✓")

                    # create conversation chain
                    st.info("Initializing conversation chain...")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Ready to chat! Ask a question below.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                    print(f"Full error: {e}")
                    import traceback
                    traceback.print_exc()
                                

    print("hello , world")


if __name__ == "__main__":
    main()