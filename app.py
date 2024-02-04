from dotenv import load_dotenv
import streamlit as st
from carebot import Carebot

# init our carebot object only once. If we don't do it this way, Streamlit
# will reinitialize the object each page load, erasing data
@st.cache_resource
def init_coachbot() -> Carebot:
    return Carebot()
bot = init_coachbot()

# list of LLM models. Add to this list to update the UI
llm_models = (
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-1106',
    'gpt-4',
    'gpt-4-1106-preview'
)

# list of embedding models, Add to this list to update the UI
embedding_models = (
    'text-embedding-ada-002',
    'more to come (dont select this)'
)

# list of vector stores, Add to this list to update the UI
vector_stores = (
    "FAISS",
    "Chroma"
)

# list of file types the user is allowed to upload to be parsed and used
# by the embedding model
#TODO 'csv'
file_types = [
    'pdf',
]

class UI(object):
    def __init__(self, title: str):
        title.lower()
        self._chunk_size = 2000
        self._overlap_size = 200
        self._temp = 0.2
        self._prompt = 'You are a therapist'

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        self._setup_sidebar()
        self._send_chat()

    def _setup_sidebar(self):
        with st.sidebar:
            st.write("Data")
            self._files = st.file_uploader(
                label="Text data",
                accept_multiple_files=True,
                type=file_types
            )
            self._chunk_size = st.number_input(
                "Chunk size", min_value=0, max_value=10000, value=self._chunk_size
            )
            self._overlap_size = st.number_input(
                "Overlap size", min_value=0, max_value=1000, value=self._overlap_size
            )
            st.write("Models")
            self._embedded = st.selectbox(
                label="Embedding model",
                options=embedding_models
            )
            self._store = st.selectbox(
                label="Vector stores",
                options=vector_stores
            )
            self._llm = st.selectbox(
                label="LLM model",
                options=llm_models
            )
            self._temp = st.slider(label='Temperature', min_value=0.0, max_value=2.0, value=self._temp)
            st.write("Prompt")
            self._prompt = st.text_area(label="Prompt", value=self._prompt)
            st.button(label="Update", on_click=self._update_coachbot)

    def _send_chat(self):
        user_input = st.chat_input(placeholder='Chatbot')
        if user_input:
            st.session_state['chat_history'].append({
                'role': 'user',
                'msg': user_input
            })
            response = bot.ask(user_input)
            st.session_state['chat_history'].append({
                'role': 'ai',
                'msg': response
            })

        for chat in st.session_state['chat_history']:
            with st.chat_message(name = chat['role']):
                st.write(chat['msg'])

    def _update_coachbot(self):
        with st.status('updating...', expanded=True) as status:
            if not self._files:
                st.error("Missing input data files, i.e PDFs.")
                return
            st.write('Loading input data...')
            bot.load_pdfs(self._files)

            st.write('Chunking data...')
            bot.chunk_data(self._chunk_size, self._overlap_size)

            if not self._embedded:
                st.error("Select an embedding model")
                return
            st.write('Creating vector store...')
            bot.create_vectorstore(self._embedded, self._store)

            if not self._prompt:
                st.error("Missing a prompt")
                return
            st.write('Setting up LLM prompt...')
            bot.create_chain(self._llm, self._temp, self._prompt)

            status.update(label="Complete", state="complete", expanded=False)
        st.toast("Update successful")

if __name__ == '__main__':
    load_dotenv()
    UI("Carebot playground")
