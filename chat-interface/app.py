import streamlit as st
from streamlit_chat import message as st_message
from streamlit_extras.colored_header import colored_header
from query_class import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil, os

EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2" # Chroma takes care if embeddings are None
LLM_FALCON_SMALL = "falcon-7b-instruct"

dir_path = ".chrome/index"
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

# Streamlit app code
st.set_page_config(
    page_title='T-Systems Chatify :)',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)

st.title("🔖 T-Systems Chatify :)")

if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

if "history" not in st.session_state:
    st.session_state.history = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi I am your assistant for TSI related works.. (If the HR's are busy then I am here to help :) "]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


#st.session_state.history.append({"message": "Hi I am your assistant for TSI related works.. (If the HR's are busy then I am here to help :) ", "is_user": False})

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):
    if llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
@st.cache_resource
def load_emb(emb):
    if emb == EMB_SBERT_MPNET_BASE:
        return PdfQA.create_sbert_mpnet()
    else:
        raise ValueError("Invalid embedding setting")
    

@st.cache_resource
def load_store():
    return PdfQA.vector_db_pdf()



with st.sidebar:
    emb, llm, load_in_8bit = EMB_SBERT_MPNET_BASE, LLM_FALCON_SMALL, False
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Uploading PDF and Generating Embeddings.."):
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                shutil.copyfileobj(pdf_file, tmp)
                tmp_path = Path(tmp.name)
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": str(tmp_path),
                    "embedding": emb,
                    "llm": llm,
                    "load_in_8bit": load_in_8bit
                }
                st.session_state["pdf_qa_model"].embedding = load_emb(emb)
                st.session_state["pdf_qa_model"].llm = load_llm(llm,load_in_8bit)        
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.sidebar.success("PDF uploaded successfully")

def generate_answer(user_input):
    st.session_state["pdf_qa_model"].retreival_qa_chain()
    #user_message = st.session_state.input_text
    answer = st.session_state["pdf_qa_model"].answer_query(user_input)
    return answer['result']
    #st.write(f"{answer['result']}")
    # st.session_state.history.append({"message": user_message, "is_user": True})
    # st.session_state.history.append({"message": answer['result'], "is_user": False})
    # st.session_state.message = ""
    # print(st.session_state.history)


def get_text():
    input_text = st.chat_input("Please let me know your doubts.. ")
    return input_text
#st.text_input('Ask a question', key="input_text", on_change=generate_answer)

with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        response = generate_answer(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            st_message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            st_message(st.session_state["generated"][i], key=str(i))

# for chat in st.session_state.history:
#     st_message(**chat) 
    
# if st.button("Answer"):
#     try:
#         st.session_state["pdf_qa_model"].retreival_qa_chain()
#         answer = st.session_state["pdf_qa_model"].answer_query(question)
#         st.write(f"{answer['result']}")

#         with st.expander('Document Similarity Search'):
#             store = st.session_state["pdf_qa_model"].vectordb
#             search = store.similarity_search_with_score(question) 
#             st.write(search[0][0].page_content) 

#     except Exception as e:
#         st.error(f"Error answering the question: {str(e)}")