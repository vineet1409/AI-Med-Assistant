import streamlit as st

import nltk
from transformers import AutoModel

from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SimpleSequentialChain
import faiss
import pandas as pd
from time import sleep

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(page_title="AI-Med-Assistant[Experiment]")

# Title and Introduction
st.title('AI Med Assistant[Experiment]: Powered by Open-Source LLMs')
st.subheader('A GenAI Application')
st.markdown('Please note that the application may take a while to load models during the initial startup. Your patience is appreciated.')

# Load custom CSS
local_css("style.css")


# Download the Punkt tokenizer for sentence splitting
@st.cache_resource
def load_models():
    nltk.download('punkt')
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.8, google_api_key="AIzaSyDatwH0wK7Iro-7J28ocINW5bbCzO-qhTk")
    
    df = pd.read_csv('med_assistant_db.csv')
    sentences = df['text'].tolist()

    faiss_index = faiss.read_index('med_assistant_index')

    return model,llm,sentences,faiss_index

def search_in_index(index, query, sentences, model, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(query_embedding_np, top_k)

    results = [(sentences[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results


def craft_prompt_generate_results(input_str, llm):
    template = """
    We are conducting an analysis to assist medical professionals by analyzing patient texts. This analysis is meant to complement, not replace, professional medical advice. Please follow the instructions below.

    As a tool designed to aid in medical diagnosis, your task is to process the provided input. The input will include a patient's description of symptoms, followed by the phrase 'End of Query'. Your analysis should identify potential medical issues based on the information provided. If the text does not indicate any concerning symptoms, state "No concerning symptoms identified." Please use your analytical capabilities to interpret the information, even if the text is disorganized.

    Input Provided:
    {input_str}

    Your analysis should include:
    - Assessment: State "No concerning symptoms identified" if the text does not suggest any medical issues.
    - Potential Disorders: List any medical disorders that the text may suggest.
    - Symptoms: Highlight specific symptoms mentioned or implied in the text.
    - Recommendations: Provide general recommendations or steps for further medical evaluation. Stress the importance of consulting a healthcare professional for an accurate diagnosis and treatment.

    Format your analysis in a clear, structured manner, using bullet points for each section.

    \n\n"""

    prompt_template = PromptTemplate(input_variables=["input_str"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(
        chains=[question_chain]
    )

    result = overall_chain.run(input_str)

    return result


if __name__ == '__main__':
    model,llm,sentences,faiss_index = load_models()
    query = st.text_area("Enter the symptoms", height=200)
    if st.button("Analyze Text"):
        if query:
            # Search in the index
            results = search_in_index(faiss_index, query, sentences, model)
            text = ''
            for sentence, score in results:
                #print(f"Sentence: {sentence}, Score: {score}")
                text+=sentence

            input_str = f"query: {query}\nEnd of Query\ntext: {text}"

            try:
                ret = craft_prompt_generate_results(input_str, llm)
                st.subheader('Response-1:\n')
                st.warning(ret)

            except Exception as e:
                st.error('Unabe to retreive, please try again later')

            sleep(5)
            try:
                ret = craft_prompt_generate_results(input_str, llm)
                st.subheader('Response-2:\n')
                st.warning(ret)

            except Exception as e:
                st.error('Unabe to retreive, please try again later')





