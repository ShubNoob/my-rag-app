import os
import sys
from dotenv import load_dotenv
import chromadb
import streamlit as st
# LangChain & NVIDIA AI Foundation Endpoints
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@st.cache_resource
def bootstrap_rag_system():
    """
    Initializes and caches the heavy-lifting RAG dependencies (LLM, Embeddings, DB, and Search).
    This function executes only ONCE on server startup and shares resources across user actions.
    """
    # 1. Load your credentials from the .env file
    load_dotenv()
    
    # Adapt API key lookup for both local environment and Streamlit Cloud secrets
    if "NVIDIA_API_KEY" in os.environ:
        api_key = os.environ["NVIDIA_API_KEY"]
    elif "NVIDIA_API_KEY" in st.secrets:
        api_key = st.secrets["NVIDIA_API_KEY"]
    else:
        print("❌ Error: NVIDIA_API_KEY is missing from your environment or Streamlit Secrets!")
        sys.exit(1)

   
    db_path = os.path.join(os.path.dirname(__file__), "..", "chroma-data")
    

    if not os.path.exists(db_path):
        print(f"❌ Error: Local database directory '{db_path}' not found.")
        print("Please verify the directory path or ensure it is deployed with your repository.")
        sys.exit(1)

    print("🤖 [CACHE OVERHEAD] Bootstrapping NVIDIA models and local vector engine...")
    
    # 3. Initialize the NVIDIA embedding engine
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", nvidia_api_key=api_key)
    
    # 4. Initialize the Nemotron-3 LLM with reasoning (thinking mode) enabled
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        nvidia_api_key=api_key,
        timeout=60,
        temperature=0.6,               
        top_p=0.95,
        max_completion_tokens=16384,               
        reasoning_budget=4096,
        chat_template_kwargs={"enable_thinking": True},
    )
    
    # 5. Connect to the local ChromaDB store
    chroma_client = chromadb.PersistentClient(path=db_path)
    vector_store = Chroma(
        client=chroma_client,
        collection_name="script_knowledge_base",
        embedding_function=embeddings,
    )
    
    # Create the retriever object to pull the top 15 closest matching context blocks
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    # 6. Initialize the web search fallback tool
    web_search = DuckDuckGoSearchRun()
    
    # Return everything as a dictionary resource to be cached
    return {
        "retriever": retriever,
        "llm": llm,
        "web_search": web_search
    }

# Global initialization: Streamlit intercepts this call after the 1st run 
# and returns the cached objects immediately.
rag_resources = bootstrap_rag_system()


def generate_script_from_rag(user_query: str):
    # 1. Load your credentials from the .env file
    load_dotenv()
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ Error: NVIDIA_API_KEY is missing from your .env file!")
        sys.exit(1)

    # 2. Verify that your local database actually exists
    if not os.path.exists("/Users/hpffilms/Desktop/OOLKA_TAKE_3/chroma-data"):
        print("❌ Error: Local database directory './chroma-data' not found.")
        print("Please make sure you have run your 'ingest_data.py' script first.")
        sys.exit(1)

    print("🤖 Bootstrapping NVIDIA models and local vector engine...")
    
    # 3. Initialize the NVIDIA embedding engine (must match the ingestion model)
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    
    # 4. Initialize the Nemotron-3 LLM with reasoning (thinking mode) enabled
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        timeout=60,
        temperature=0.6,               # Slightly lowered for more structural consistency
        top_p=0.95,
        max_completion_tokens=16384,               
        reasoning_budget=4096,
        chat_template_kwargs={"enable_thinking": True},
    )
    
    # 5. Connect to the local ChromaDB store
    chroma_client = chromadb.PersistentClient(path="/Users/hpffilms/Desktop/OOLKA_TAKE_3/chroma-data")
    vector_store = Chroma(
        client=chroma_client,
        collection_name="script_knowledge_base",
        embedding_function=embeddings,
    )
    
    # Create the retriever object to pull the top 3 closest matching context blocks
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    # 6. Initialize an optional web search fallback tool for live trends
    web_search = DuckDuckGoSearchRun()

    # 7. Step A: Perform the semantic search to gather internal documents
    print(f"🔍 [STEP 1/3] Retrieving matching chunks for: \"{user_query}\"")
    retrieved_docs = retriever.invoke(user_query)
    
    if not retrieved_docs:
        print("⚠️ No direct database matches found. Proceeding with empty internal context.")
        internal_context = "No specific internal company guidelines or historical scripts matched this exact query."
    else:
        print(f"✅ Found {len(retrieved_docs)} relevant context chunks.")
        # Flatten the text chunks into a readable text block for the LLM prompt
        internal_context = "\n---\n".join([doc.page_content.strip() for doc in retrieved_docs])

    # 8. Step B: Perform web search to gather live hooks and trend analysis
    print("🌐 [STEP 2/3] Fetching live internet trends via DuckDuckGo...")
    search_query = f"most engaging video ideas for {user_query}"
    try:
        web_context = web_search.run(search_query)
    except Exception as e:
        print(f"⚠️ Web search timed out or failed ({e}). Defaulting to database context only.")
        web_context = "No external live trends available at this moment."

    # 9. Step C: Build the Master Prompt Template
    # This guides the model to reason precisely across your data and output a predictable format.
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

    # 10. Assemble the complete LangChain pipeline (Prompt -> LLM -> String Output Parser)
    chain = prompt_template | llm | StrOutputParser()
    
    print("🧠 [STEP 3/3] Assembling context. Calling Nemotron-3 reasoning engine...")
    print("Please wait while the model evaluates the best approach...")
    print("-" * 60)

    # Execute the chain
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
    
    # Input your script target prompt directly here
    target_prompt = "lets write a skit based oolka script where the setting is front of temple and there are 2 chareceters one is beggar and one is a buswiness man and the beggar teaches about his regret to the business man and tells him to use oolka before its too late"
    script_output = generate_script_from_rag(target_prompt)
    
    print("\n================= GENERATED SCRIPT =================")
    print(script_output)
    print("====================================================")

