#!/usr/bin/env python3
"""
AmbedkarGPT - A simple Q&A system using RAG (Retrieval-Augmented Generation)
This system ingests text from Dr. B.R. Ambedkar's speech and answers questions based on that content.
"""

# Imports
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

class AmbedkarGPT:
    def __init__(self, speech_file="speech.txt", persist_directory="./chroma_db"):
        """Initialize the Q&A system with the speech file."""
        self.speech_file = speech_file
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
    def load_and_split_text(self):
        """Load the speech text and split it into manageable chunks."""
        print("Loading and splitting text...")
        
        # Load the text file
        loader = TextLoader(self.speech_file, encoding='utf-8')
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        print(f"✓ Text split into {len(texts)} chunks")
        return texts
    
    def create_embeddings(self):
        """Create embeddings using HuggingFace sentence-transformers."""
        if self.embeddings is None:
            print("Loading embeddings model...")
            
            # Initialize HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("Embeddings model loaded successfully")
        return self.embeddings
    
    def load_or_create_vector_store(self):
        """Load existing vector store or create new one if it doesn't exist."""
        # Create embeddings first
        self.create_embeddings()
        
        # Check if vector store already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading cached vector store from disk")
            
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            print("Vector store loaded successfully")
        else:
            print("No cached vector store found. Creating new vector store and caching for potential future use...")
            
            # Load and split text
            texts = self.load_and_split_text()
            
            # Create new ChromaDB vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print("Vector store created and saved to disk")
        
        return self.vectorstore
    
    def setup_qa_chain(self):
        """Set up the RetrievalQA chain with CPU-optimized Ollama LLM."""
        print("Setting up QA chain with CPU-optimized settings...")
        
        # Better performance for CPU-only systems, since I don't have a dedicated GPU :(
        llm = OllamaLLM(
            model="mistral:7b-instruct-q4_0",
            temperature=0.7,
            #num_ctx=2048,
            num_thread=None,
        )
        
        # Create a custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Retriever creation
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Creating RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("QA chain setup complete")
        return self.qa_chain
    
    def initialize_system(self):
        """Initialize the RAG pipeline"""
        print("\n" + "=" * 60)
        print("Initializing AmbedkarGPT system...")
        print("=" * 60 + "\n")
        
        # Load or create vector store
        self.load_or_create_vector_store()
        
        # Setup QA chain
        self.setup_qa_chain()
        
        print("\nInitialization complete!\n")
    
    def ask_question(self, question):
        """Ask a question and get an answer based on the speech content without hallucinations"""
        if not self.qa_chain:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        print(f"\nQuestion(or Choice): {question}")
        print("Searching for relevant information...")
        
        # Calling Langchain pipeline
        result = self.qa_chain.invoke({"query": question})
        
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        print(f"\nAnswer: {answer}")
        print(f"\nBased on {len(source_docs)} relevant text chunks from the speech.")
        
        return answer, source_docs
    
    def reset_vector_store(self):
        """Delete existing vector store to force recreation (useful if speech.txt changed)."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("Vector store deleted. It will be recreated on next run.")


def main():
    """Main function to run the interactive Q&A system."""
    print("\n" + "=" * 70)
    print("Welcome to the CLI Q&A System - Dr. B.R. Ambedkar Speech")
    print("=" * 70)
    
    # Initialize the system
    ambedkar_gpt = AmbedkarGPT()
    
    try:
        ambedkar_gpt.initialize_system()
    except FileNotFoundError as e:
        print(f"\nError: speech.txt file not found!")
        print("Please ensure speech.txt exists in the current directory.")
        return
    except Exception as e:
        print(f"\nError initializing system: {e}")
        print("\nPlease ensure:")
        print("   1. Ollama is running (try: ollama serve)")
        print("   2. Mistral model is available (try: ollama pull mistral)")
        print("   3. All dependencies are installed (try: pip install -r requirements.txt)")
        return
    
    print("=" * 70)
    print("System ready for questions! You can now ask questions about Dr. Ambedkar's speech.")
    print("Type 'quit', 'exit', or 'q' to stop the program, and press Enter to confirm your choices.")
    print("Type 'reset' to delete cached vector store (if you changed speech.txt)")
    print("=" * 70)
    
    # Q&A Loop
    while True:
        try:
            question = input("\nYour question/option: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using AmbedkarGPT! Goodbye!")
                break
            
            if question.lower() == 'reset':
                ambedkar_gpt.reset_vector_store()
                print("Please restart the program to recreate the vector store.")
                continue
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            # Get answer
            answer, sources = ambedkar_gpt.ask_question(question)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing question: {e}")
            print("Please try again or check if Ollama is running (Try running 'ollama serve' in a separate terminal or start the ollama service).")


if __name__ == "__main__":
    main()