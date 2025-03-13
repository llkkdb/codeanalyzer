import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import click
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI  # Or your preferred LLM

class CodeUnderstandingSystem:
    def __init__(self):
        self.context_files = []
        self.vector_store = None
        self.llm = ChatOpenAI(temperature=0.1)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.knowledge_guidelines = self._load_knowledge_guidelines()

    def _load_knowledge_guidelines(self) -> str:
        """Load knowledge guidelines from file"""
        try:
            with open("knowledge_guidelines.md", "r") as f:
                return f.read()
        except FileNotFoundError:
            return "# Codebase Knowledge Guidelines\n\n1. No specific guidelines available"

    def generate_search_commands(self, query: str) -> List[str]:
        """Use LLM to generate search commands based on knowledge guidelines"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.knowledge_guidelines + "\nGenerate Linux find/grep commands to locate relevant code. Respond only with commands separated by newlines."),
            ("human", "Find code related to: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        return [cmd.strip() for cmd in response.content.split("\n") if cmd.strip()]

    def execute_search(self, commands: List[str]) -> List[str]:
        """Execute search commands and collect files"""
        found_files = set()
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd, shell=True, check=True,
                    stdout=subprocess.PIPE, text=True
                )
                for line in result.stdout.split("\n"):
                    if line.strip() and Path(line.strip()).exists():
                        found_files.add(line.strip())
            except subprocess.CalledProcessError:
                continue
                
        return list(found_files)

    def add_to_context(self, files: List[str], chunk_size=1000) -> None:
        """Add files to RAG context with chunking"""
        docs = []
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Simple chunking
                chunks = [content[i:i+chunk_size] 
                          for i in range(0, len(content), chunk_size)]
                for i, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk": i+1
                        }
                    ))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if docs:
            if self.vector_store:
                self.vector_store.add_documents(docs)
            else:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def answer_with_context(self, query: str) -> str:
        """Use RAG context to answer question"""
        if not self.vector_store:
            return "No context available"
            
        relevant = self.vector_store.similarity_search(query, k=5)
        context = "\n\n".join(
            f"File: {doc.metadata['source']} (Chunk {doc.metadata['chunk']})\n"
            f"{doc.page_content[:500]}..." 
            for doc in relevant
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.knowledge_guidelines + "\nAnswer using this context:"),
            ("system", "Context:\n{context}"),
            ("human", "Question: {query}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({"context": context, "query": query}).content

@click.group()
@click.pass_context
def cli(ctx):
    """LLM-Guided Code Understanding System"""
    ctx.obj = CodeUnderstandingSystem()

@cli.command()
@click.argument("query")
@click.pass_obj
def ask(system, query):
    """Ask a question about the codebase"""
    # Step 1: Generate search commands using knowledge guidelines
    commands = system.generate_search_commands(query)
    click.echo(f"Generated search commands:\n" + "\n".join(commands))
    
    # Step 2: Execute searches
    found_files = system.execute_search(commands)
    click.echo(f"\nFound {len(found_files)} files")
    
    # Step 3: Add to RAG context
    system.add_to_context(found_files)
    
    # Step 4: Generate answer using context
    answer = system.answer_with_context(query)
    click.echo(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    cli()