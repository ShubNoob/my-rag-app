import os
import sys
from dotenv import load_dotenv
import chromadb
import streamlit as st

# LangChain Ecosystem Importers
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@st.cache_resource
def bootstrap_rag_system():
    """
    Initializes and caches the hybrid RAG dependencies.
    Uses NVIDIA for vector embeddings and Anthropic Claude for script generation.
    """
    load_dotenv()
    
    # 1. Fetch NVIDIA API Key (For Ingestion/Database Reading)
    if "NVIDIA_API_KEY" in os.environ:
        nvidia_key = os.environ["NVIDIA_API_KEY"]
    elif "NVIDIA_API_KEY" in st.secrets:
        nvidia_key = st.secrets["NVIDIA_API_KEY"]
    else:
        print("❌ Error: NVIDIA_API_KEY is missing from Environment or Streamlit Secrets!")
        sys.exit(1)

    # 2. Fetch Anthropic API Key (For Claude Generation)
    if "ANTHROPIC_API_KEY" in os.environ:
        anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    elif "ANTHROPIC_API_KEY" in st.secrets:
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    else:
        print("❌ Error: ANTHROPIC_API_KEY is missing from Environment or Streamlit Secrets!")
        sys.exit(1)

    # 3. Setup Relative Database Path
    db_path = os.path.join(os.path.dirname(__file__), "chroma-data")

    if not os.path.exists(db_path):
        print(f"❌ Error: Local database directory '{db_path}' not found.")
        sys.exit(1)

    print("🤖 [CACHE OVERHEAD] Bootstrapping hybrid NVIDIA-Embeddings & Claude-LLM system...")
    
    # 4. Initialize NVIDIA Embeddings (Must match your original ingestion model)
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", nvidia_api_key=nvidia_key)
    
    # 5. Initialize Claude LLM Endpoint
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",  
        anthropic_api_key=anthropic_key,
        temperature=0.6,                
        max_tokens=8192,                
    )
    
    # 6. Connect to the local ChromaDB store
    chroma_client = chromadb.PersistentClient(path=db_path)
    vector_store = Chroma(
        client=chroma_client,
        collection_name="script_knowledge_base",
        embedding_function=embeddings,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    web_search = DuckDuckGoSearchRun()
    
    return {
        "retriever": retriever,
        "llm": llm,
        "web_search": web_search
    }


def generate_script_from_rag(user_query: str):
    """
    Main runner function. Extracts tools from the global cache resource
    without blocking the Streamlit UI initialization phase.
    """
    rag_resources = bootstrap_rag_system()
    retriever = rag_resources["retriever"]
    llm = rag_resources["llm"]
    web_search = rag_resources["web_search"]

    # 1. Vector Search Context Gathering
    print(f"🔍 [STEP 1/3] Retrieving matching chunks for: \"{user_query}\"")
    retrieved_docs = retriever.invoke(user_query)
    
    if not retrieved_docs:
        print("⚠️ No direct database matches found. Proceeding with empty internal context.")
        internal_context = "No specific internal company guidelines or historical scripts matched this exact query."
    else:
        print(f"✅ Found {len(retrieved_docs)} relevant context chunks.")
        internal_context = "\n---\n".join([doc.page_content.strip() for doc in retrieved_docs])

    # 2. Live Web Hook Fetching
    print("🌐 [STEP 2/3] Fetching live internet trends via DuckDuckGo...")
    search_query = f"most engaging video ideas for {user_query}"
    try:
        web_context = web_search.run(search_query)
    except Exception as e:
        print(f"⚠️ Web search timed out or failed ({e}). Defaulting to database context only.")
        web_context = "No external live trends available at this moment."

    # 3. Master Prompt Execution
    template = """You are an elite scriptwriter who specializes in high-retention verbal storytelling. Your core objective is to write incredibly engaging text and dialogues. 
    
    Minimize deep focus on visual execution, complex SFX, or BGM. Instead, maximize your focus on the verbal delivery, rhythm, emotional texture of the dialogue, and conversational phrasing.

    ---
    CRITICAL WRITING STYLES & DIALOGUE CONSTRAINTS:
    - EMBRACE HUMAN IMPERFECTIONS: Write exactly how real people talk. Use short, blunt, or uneven sentences. Use colloquial transitions, sentence fragments, and natural contractions (it's, don't, you're, gonna).
    - COHESIVE PACING: Ensure a rhythmic verbal flow. Read the script internally to make sure a speaker can deliver it naturally without running out of breath or sounding like a rigid infomercial.

    ---
    REQUIRED SCRIPT ARCHITECTURE (Strict 5-Pillar Layout):

    1. [HOOK] (0-5 Seconds)
       - Open with an immediate, counter-intuitive verbal statement or an raw emotional friction point.
       - No introductions, no pleasantries, and no brand mentions. Shock or deeply hook the viewer with the very first line spoken.
       - Keep visual notes minimal; prioritize the raw opening dialogue.

    2. [STORY]
       - The story should be based on the hook and it should be relatable to the viewer.
       - The story should be engaging and it should keep the viewer hooked.

    3. [BRIDGE] (Connecting the story with out product)
       - Seamlessly transition from the narrative to the product introduction in a conversational and engaging manner.
       - Highlight how the product addresses the pain point and delivers value.

    4. [PRODUCT] (The Resolution & Call to Action)
       - Seamlessly bridge the dialogue into the specific solution or feature set without dropping into a generic sales pitch tone.
       - Conclude with a low-friction, high-intent verbal Call to Action (CTA) that sounds like an organic recommendation from a peer.

    ---
    CONTEXTUAL BOUNDARIES:
    
    INTERNAL ASSETS & SOURCE DIRECTIVES:
    {internal_context}

    EXTERNAL LIVE INTERNET TRENDS & VIRAL HOOK DATA:
    {web_context}

    ---
    USER SPECIFIC TASK AND DIRECTIVE:
    {question}

    Synthesize all assets and write the definitive dialogue-driven production script now:
    """
    
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()
    
    print("🧠 [STEP 3/3] Assembling context. Calling Claude...")
    print("-" * 60)

    final_script = chain.invoke({
        "internal_context": internal_context,
        "web_context": web_context,
        "question": user_query
    })
    
    return final_script


if __name__ == "__main__":
    print("==================================================")
    print("🎬 Welcome to the RAG Script Generation Pipeline!")
    print("==================================================\n")
    
    target_prompt = "lets write a skit based oolka script where the setting is front of temple and there are 2 chareceters one is beggar and one is a business man and the beggar teaches about his regret to the business man and tells him to use oolka before its too late"
    script_output = generate_script_from_rag(target_prompt)
    
    print("\n================= GENERATED SCRIPT =================")
    print(script_output)
    print("====================================================")
