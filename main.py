#!/usr/bin/env python3
"""
RAG-Enhanced Scientific Research Assistant
Main executable application
"""

import asyncio
import argparse
import sys
import os
from typing import List, Optional
from loguru import logger
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.rag_pipeline import RAGPipeline, SearchResult, RAGResponse
from config.config import config

class ResearchAssistantCLI:
    """Command-line interface for the Research Assistant"""
    
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
        logger.add(
            f"{config.database.LOG_DIR}/research_assistant.log",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
    
    def print_banner(self):
        """Print application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   RAG-Enhanced Scientific Research Assistant                  ║
║                    AI/ML Literature Analysis & Hypothesis Generation          ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_help(self):
        """Print available commands"""
        help_text = """
Available Commands:
─────────────────────────────────────────────────────────────────────────────
  search <query>              - Search literature for a specific topic
  recent <category> [days]     - Get recent papers from a category (default: 7 days)
  full <query>                 - Complete analysis (search + summary + hypotheses)
  sources                      - List available literature sources
  categories                   - List available paper categories
  export <filename>            - Export last results to JSON file
  config                       - Show current configuration
  help                         - Show this help message
  exit                         - Exit the application

Examples:
─────────────────────────────────────────────────────────────────────────────
  > search "transformer attention mechanisms"
  > recent cs.AI 14
  > full "reinforcement learning for robotics"
  > export research_results.json
        """
        print(help_text)
    
    def format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for display"""
        if not results:
            return "No papers found."
        
        formatted = f"\n📚 Found {len(results)} papers:\n" + "="*80 + "\n"
        
        for i, paper in enumerate(results[:10], 1):  # Show top 10
            formatted += f"\n{i}. {paper.title}\n"
            formatted += f"   Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                formatted += f" (+{len(paper.authors)-3} more)"
            formatted += f"\n   Source: {paper.source.upper()}"
            if paper.citation_count:
                formatted += f" | Citations: {paper.citation_count}"
            if paper.published_date:
                formatted += f" | Published: {paper.published_date[:10]}"
            formatted += f" | Relevance: {paper.relevance_score:.3f}"
            formatted += f"\n   URL: {paper.url}"
            formatted += f"\n   Abstract: {paper.abstract[:200]}..."
            formatted += "\n" + "-"*80
        
        if len(results) > 10:
            formatted += f"\n... and {len(results)-10} more papers\n"
        
        return formatted
    
    def format_response(self, response: RAGResponse) -> str:
        """Format complete RAG response for display"""
        formatted = f"\n🔍 Query: {response.query}\n"
        formatted += "="*80 + "\n"
        
        # Papers
        formatted += self.format_search_results(response.retrieved_papers)
        
        # Summary
        formatted += f"\n📝 Summary:\n{'-'*40}\n"
        formatted += response.summary + "\n"
        
        # Hypotheses
        if response.hypotheses:
            formatted += f"\n💡 Research Hypotheses:\n{'-'*40}\n"
            for i, hypothesis in enumerate(response.hypotheses, 1):
                formatted += f"{i}. {hypothesis}\n"
        
        # Metadata
        formatted += f"\n📊 Analysis Metadata:\n{'-'*40}\n"
        formatted += f"Papers retrieved: {response.metadata.get('num_papers_retrieved', 0)}\n"
        formatted += f"Sources used: {', '.join(response.metadata.get('sources_used', []))}\n"
        formatted += f"Processing time: {response.metadata.get('processing_time_seconds', 0):.2f} seconds\n"
        formatted += f"Average relevance: {response.metadata.get('average_relevance_score', 0):.3f}\n"
        
        return formatted
    
    async def handle_search(self, query: str, sources: List[str] = None):
        """Handle search command"""
        logger.info(f"Searching for: {query}")
        results = await self.pipeline.search_literature(query, sources)
        print(self.format_search_results(results))
        self.last_results = results
        return results
    
    async def handle_full_analysis(self, query: str, sources: List[str] = None):
        """Handle full analysis command"""
        logger.info(f"Performing full analysis for: {query}")
        response = await self.pipeline.process_query(query, sources)
        print(self.format_response(response))
        self.last_response = response
        return response
    
    async def handle_recent_papers(self, category: str = "cs.AI", days: int = 7):
        """Handle recent papers command"""
        logger.info(f"Getting recent papers from {category} (last {days} days)")
        results = await self.pipeline.arxiv_client.get_recent_papers(category, days)
        
        # Convert to SearchResult format
        search_results = []
        for paper in results:
            search_result = SearchResult(
                id=paper.id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                source='arxiv',
                url=paper.url,
                published_date=paper.published,
                categories=paper.categories,
                doi=paper.doi
            )
            search_results.append(search_result)
        
        print(self.format_search_results(search_results))
        self.last_results = search_results
        return search_results
    
    def handle_export(self, filename: str):
        """Export last results to JSON file"""
        if not hasattr(self, 'last_response') and not hasattr(self, 'last_results'):
            print("❌ No results to export. Run a search first.")
            return
        
        try:
            data = {}
            if hasattr(self, 'last_response'):
                # Export complete response
                data = {
                    'query': self.last_response.query,
                    'papers': [
                        {
                            'id': p.id,
                            'title': p.title,
                            'authors': p.authors,
                            'abstract': p.abstract,
                            'source': p.source,
                            'url': p.url,
                            'published_date': p.published_date,
                            'citation_count': p.citation_count,
                            'venue': p.venue,
                            'categories': p.categories,
                            'doi': p.doi,
                            'relevance_score': p.relevance_score
                        } for p in self.last_response.retrieved_papers
                    ],
                    'summary': self.last_response.summary,
                    'hypotheses': self.last_response.hypotheses,
                    'metadata': self.last_response.metadata,
                    'timestamp': self.last_response.timestamp
                }
            elif hasattr(self, 'last_results'):
                # Export search results only
                data = {
                    'papers': [
                        {
                            'id': p.id,
                            'title': p.title,
                            'authors': p.authors,
                            'abstract': p.abstract,
                            'source': p.source,
                            'url': p.url,
                            'published_date': p.published_date,
                            'citation_count': p.citation_count,
                            'venue': p.venue,
                            'categories': p.categories,
                            'doi': p.doi,
                            'relevance_score': p.relevance_score
                        } for p in self.last_results
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results exported to {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def show_sources(self):
        """Show available literature sources"""
        sources_info = """
📚 Available Literature Sources:
─────────────────────────────────────────────────────────────────────────────
  arXiv           - Preprint repository for physics, mathematics, CS, etc.
  Semantic Scholar - Academic paper search engine with citation analysis
  PubMed          - Biomedical and life science literature (coming soon)

🔧 Source Selection:
  Default: arxiv, semantic_scholar
  Usage: Add --sources arxiv semantic_scholar to commands
        """
        print(sources_info)
    
    def show_categories(self):
        """Show available paper categories"""
        categories_info = """
📂 arXiv Categories:
─────────────────────────────────────────────────────────────────────────────
  cs.AI     - Artificial Intelligence      cs.LG     - Machine Learning
  cs.CL     - Computation and Language     cs.CV     - Computer Vision
  cs.RO     - Robotics                     cs.NE     - Neural Networks
  stat.ML   - Machine Learning (Stats)     math.OC   - Optimization
  
📊 Semantic Scholar Fields:
─────────────────────────────────────────────────────────────────────────────
  Computer Science    Mathematics         Physics
  Biology            Chemistry           Medicine
  Psychology         Economics           Engineering
        """
        print(categories_info)
    
    def show_config(self):
        """Show current configuration"""
        config_info = f"""
⚙️  Current Configuration:
─────────────────────────────────────────────────────────────────────────────
  Device: {config.model.DEVICE}
  Max Results: {config.rag.MAX_RETRIEVED_DOCS}
  Embedding Model: {config.model.SCIBERT_MODEL}
  Vector Index: {config.model.VECTOR_INDEX_TYPE}
  
  API Settings:
  - arXiv Rate Limit: {config.api.ARXIV_RATE_LIMIT} req/sec
  - Semantic Scholar Rate Limit: {config.api.SEMANTIC_SCHOLAR_RATE_LIMIT} req/sec
  - PubMed API Key: {'✅ Set' if config.api.PUBMED_API_KEY else '❌ Not set'}
  - Semantic Scholar API Key: {'✅ Set' if config.api.SEMANTIC_SCHOLAR_API_KEY else '❌ Not set'}
        """
        print(config_info)
    
    async def run_interactive(self):
        """Run interactive CLI session"""
        self.print_banner()
        print("🤖 Welcome to the RAG-Enhanced Scientific Research Assistant!")
        print("Type 'help' for available commands or 'exit' to quit.\n")
        
        while True:
            try:
                command = input("🔬 > ").strip()
                
                if not command:
                    continue
                
                if command.lower() == 'exit':
                    print("👋 Thank you for using the Research Assistant!")
                    break
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'help':
                    self.print_help()
                
                elif cmd == 'search':
                    if len(parts) < 2:
                        print("❌ Usage: search <query>")
                        continue
                    query = ' '.join(parts[1:])
                    await self.handle_search(query)
                
                elif cmd == 'full':
                    if len(parts) < 2:
                        print("❌ Usage: full <query>")
                        continue
                    query = ' '.join(parts[1:])
                    await self.handle_full_analysis(query)
                
                elif cmd == 'recent':
                    category = parts[1] if len(parts) > 1 else "cs.AI"
                    days = int(parts[2]) if len(parts) > 2 else 7
                    await self.handle_recent_papers(category, days)
                
                elif cmd == 'export':
                    filename = parts[1] if len(parts) > 1 else f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.handle_export(filename)
                
                elif cmd == 'sources':
                    self.show_sources()
                
                elif cmd == 'categories':
                    self.show_categories()
                
                elif cmd == 'config':
                    self.show_config()
                
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                
                print()  # Add spacing between commands
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG-Enhanced Scientific Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['search', 'recent', 'full', 'interactive'],
        default='interactive',
        help='Command to run (default: interactive)'
    )
    
    parser.add_argument(
        'query',
        nargs='*',
        help='Search query or parameters'
    )
    
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['arxiv', 'semantic_scholar', 'pubmed'],
        default=['arxiv', 'semantic_scholar'],
        help='Literature sources to search'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    cli = ResearchAssistantCLI()
    
    async def run_command():
        if args.command == 'interactive' or not args.query:
            await cli.run_interactive()
        else:
            query = ' '.join(args.query)
            
            if args.command == 'search':
                results = await cli.handle_search(query, args.sources)
                if args.output:
                    cli.last_results = results
                    cli.handle_export(args.output)
            
            elif args.command == 'full':
                response = await cli.handle_full_analysis(query, args.sources)
                if args.output:
                    cli.last_response = response
                    cli.handle_export(args.output)
            
            elif args.command == 'recent':
                category = args.query[0] if args.query else "cs.AI"
                days = int(args.query[1]) if len(args.query) > 1 else 7
                results = await cli.handle_recent_papers(category, days)
                if args.output:
                    cli.last_results = results
                    cli.handle_export(args.output)
    
    try:
        asyncio.run(run_command())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 