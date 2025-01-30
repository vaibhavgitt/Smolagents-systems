

import os
import json
import gradio as gr
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from typing import List, Dict

from smolagents import (
    CodeAgent,
    VisitWebpageTool,
    HfApiModel,
    ToolCallingAgent,
    DuckDuckGoSearchTool
)

class MultiAgentDocumentationScraper:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        # Initialize agents
        self.web_search_agent = ToolCallingAgent(
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=HfApiModel(),
            max_steps=10
        )

        self.doc_extraction_agent = CodeAgent(
            tools=[VisitWebpageTool()],
            model=HfApiModel(),
            additional_authorized_imports=['re', 'json', 'requests']
        )

        # Embedding setup
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)

        # Initialize ChromaDB with new client API
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Create or load collection with proper configuration
        self.collection = self.chroma_client.get_or_create_collection(
            name="documentation_store",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # In-memory store for metadata
        self.documentation_metadata = []

    def _generate_embedding(self, text):
        """Generate embedding for a text string"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling to get fixed-size embedding
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

    def scrape_library_documentation(self, library_name, query):
        """Multi-agent documentation scraping with vector indexing"""
        # Web search phase
        search_query = f"{library_name} {query} official documentation"
        search_results = self.web_search_agent.run(search_query)

        # Documentation extraction
        documentation = []
        for result in search_results[:3]:
            extraction_prompt = f"""
            Extract comprehensive documentation from {result} focusing on:
            - {query}
            - Related modules
            - Code examples
            - API references
            """
            doc_content = self.doc_extraction_agent.run(extraction_prompt)
            documentation.append(doc_content)

        # Process and store results
        combined_doc = "\n\n".join(documentation)
        doc_id = str(len(self.documentation_metadata))

        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._generate_embedding(combined_doc)],
            metadatas=[{
                "library": library_name,
                "query": query,
                "sources": search_results[:3],
                "timestamp": datetime.now().isoformat()
            }],
            documents=[combined_doc]
        )

        # Store metadata
        self.documentation_metadata.append({
            "id": doc_id,
            "library": library_name,
            "query": query,
            "content": combined_doc,
            "sources": search_results[:3]
        })

        return combined_doc

    def search_documentation(self, query, top_k=3):
        """Semantic search through documentation using ChromaDB"""
        if len(self.documentation_metadata) == 0:
            return "No documentation available yet!"

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        # Format results
        formatted_results = []
        for i, (metadata, doc, distance) in enumerate(zip(
            results['metadatas'][0],
            results['documents'][0],
            results['distances'][0]
        )):
            formatted_results.append(
                f"### Result {i+1}\n"
                f"**Library**: {metadata['library']}\n"
                f"**Relevance Score**: {1 - distance:.2f}\n"
                f"**Content**:\n{doc[:1000]}...\n"
                f"**Sources**: {', '.join(metadata['sources'])}"
            )

        return "\n\n".join(formatted_results) if formatted_results else "No relevant documentation found."

    def export_to_json(self, filename=None):
        """Export documentation store to JSON file"""
        if not filename:
            filename = f"documentation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(self.documentation_metadata, f, indent=2)

        return filename

    def clear_documentation(self):
        """Clear all stored documentation"""
        self.collection.delete()
        self.documentation_metadata.clear()
        return "Documentation store cleared successfully!"

def create_gradio_interface(scraper):
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        with gr.Tab("Documentation Scraper"):
            gr.Markdown("## Multi-Agent Documentation Scraper")
            with gr.Row():
                library_input = gr.Textbox(label="Library/Framework Name")
                query_input = gr.Textbox(label="Search Query")

            with gr.Row():
                submit_btn = gr.Button("Search", variant="primary")
                export_btn = gr.Button("Export to JSON", variant="secondary")
                clear_btn = gr.Button("Clear Inputs", variant="stop")

            doc_output = gr.Markdown("### Scraped Documentation")
            json_output = gr.File(label="Exported JSON")

        with gr.Tab("Documentation Q&A"):
            gr.Markdown("## Semantic Documentation Search")
            search_input = gr.Textbox(label="Search Query", placeholder="Enter your question...")
            search_btn = gr.Button("Search Documentation")
            search_output = gr.Markdown("### Search Results")

        # Scraping functionality
        submit_btn.click(
            scraper.scrape_library_documentation,
            inputs=[library_input, query_input],
            outputs=doc_output
        )

        # Export functionality
        export_btn.click(
            scraper.export_to_json,
            outputs=json_output
        )

        # Search functionality
        search_btn.click(
            scraper.search_documentation,
            inputs=search_input,
            outputs=search_output
        )

        # Clear functionality - only clears inputs, preserves documentation
        clear_btn.click(
            lambda: ("", "", ""),  # Clear library, query, and search inputs
            outputs=[library_input, query_input, search_input]
        )

    return interface

def main():
    scraper = MultiAgentDocumentationScraper()
    interface = create_gradio_interface(scraper)
    interface.launch(debug=True)

if __name__ == "__main__":
    main()
