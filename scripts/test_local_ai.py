#!/usr/bin/env python3
"""
Local AI Model Testing and Demonstration Script
Test your custom trained AI model and compare with external APIs
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_banner():
    """Print test banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                      🧪 Local AI Model Testing Suite                          ║
║                                                                                ║
║  Test and compare your custom AI model with external APIs                     ║
║  Validate model performance and integration                                    ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def test_local_model_standalone(model_path: str):
    """Test local model in standalone mode"""
    print(f"\n🤖 Testing Local AI Model: {Path(model_path).name}")
    print("=" * 60)
    
    try:
        from models.local_ai_model import LocalAIInference
        
        # Load model
        print("Loading model...")
        model = LocalAIInference(model_path)
        print("✅ Model loaded successfully")
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "deep neural networks",
            "computer vision applications",
            "natural language processing",
            "reinforcement learning"
        ]
        
        # Sample context from database
        from database.paper_db import PaperDatabase
        db = PaperDatabase()
        papers = db.get_all_papers()[:3]
        
        if not papers:
            print("❌ No papers in database for testing")
            return
        
        sample_context = papers[0].abstract if papers else "Sample research context"
        
        print(f"\n📊 Model Performance Tests:")
        print("-" * 40)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            start_time = time.time()
            
            # Test relevance scoring
            try:
                relevance = model.calculate_relevance(query, sample_context)
                print(f"   Relevance Score: {relevance:.2f}/5.0")
            except Exception as e:
                print(f"   Relevance Error: {e}")
            
            # Test summary generation
            try:
                summary = model.generate_summary(query, sample_context, max_length=100)
                print(f"   Summary: {summary[:80]}...")
            except Exception as e:
                print(f"   Summary Error: {e}")
            
            # Test keyword extraction
            try:
                keywords = model.extract_keywords(sample_context, top_k=3)
                print(f"   Keywords: {', '.join(keywords)}")
            except Exception as e:
                print(f"   Keywords Error: {e}")
            
            response_time = time.time() - start_time
            print(f"   Response Time: {response_time:.2f}s")
        
        print(f"\n✅ Standalone model testing completed")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")

async def test_integrated_rag_system():
    """Test the integrated RAG system with local AI"""
    print(f"\n🔧 Testing Integrated RAG System")
    print("=" * 60)
    
    try:
        # Test with local AI model
        from rag.local_ai_enhanced_rag import LocalAIEnhancedRAG
        
        # Find local model
        model_dir = Path("data/models/local_ai")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                model_path = str(model_files[0])
                print(f"Using model: {Path(model_path).name}")
                
                # Initialize RAG with local model
                rag_local = LocalAIEnhancedRAG(use_local_model=True, local_model_path=model_path)
                
                # Get model status
                status = rag_local.get_model_status()
                print(f"\n📊 Model Status:")
                print(f"   Local AI Available: {status['local_ai_available']}")
                print(f"   Using Local Model: {status['use_local_model']}")
                print(f"   Enhanced Search: {status['enhanced_search']}")
                print(f"   Indexed Papers: {status['indexed_papers']}")
                print(f"   Cache Size: {status['search_cache_size']}")
                
                if status['local_ai_available']:
                    print(f"   Model Functional: {status.get('local_model_functional', 'Unknown')}")
                
                # Test query
                test_query = "machine learning for computer vision"
                print(f"\n🔍 Testing Query: '{test_query}'")
                
                start_time = time.time()
                analysis = await rag_local.research_and_analyze(test_query)
                end_time = time.time()
                
                print(f"\n📄 Analysis Results:")
                print(f"   Papers Found: {analysis.processing_metadata['papers_found']}")
                print(f"   Processing Time: {end_time - start_time:.2f}s")
                print(f"   Model Used: {analysis.processing_metadata['llm_model_used']}")
                print(f"   Search Type: {analysis.processing_metadata['search_type']}")
                
                if analysis.papers:
                    print(f"\n📚 Sample Papers:")
                    for i, paper in enumerate(analysis.papers[:2], 1):
                        print(f"   {i}. {paper.title[:60]}...")
                
                print(f"\n📝 Summary Preview:")
                summary_preview = analysis.summarized_findings[:200] + "..." if len(analysis.summarized_findings) > 200 else analysis.summarized_findings
                print(f"   {summary_preview}")
                
                print(f"\n✅ Integrated RAG testing completed")
                
            else:
                print("❌ No trained models found. Train a model first with:")
                print("   python scripts/train_local_model.py")
        else:
            print("❌ Model directory not found. Train a model first.")
            
    except Exception as e:
        print(f"❌ Integrated RAG testing failed: {e}")

async def compare_models():
    """Compare local AI model with external APIs"""
    print(f"\n⚖️  Model Comparison: Local AI vs External APIs")
    print("=" * 60)
    
    try:
        from rag.local_ai_enhanced_rag import LocalAIEnhancedRAG
        
        # Find local model
        model_dir = Path("data/models/local_ai")
        local_model_available = False
        
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                local_model_available = True
                model_path = str(model_files[0])
        
        test_queries = [
            "deep learning applications",
            "neural network architectures"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)
            
            # Test Local AI Model
            if local_model_available:
                print("🤖 Local AI Model:")
                try:
                    rag_local = LocalAIEnhancedRAG(use_local_model=True, local_model_path=model_path)
                    
                    start_time = time.time()
                    analysis_local = await rag_local.research_and_analyze(query)
                    local_time = time.time() - start_time
                    
                    print(f"   Response Time: {local_time:.2f}s")
                    print(f"   Papers Found: {analysis_local.processing_metadata['papers_found']}")
                    print(f"   Model: {analysis_local.processing_metadata['llm_model_used']}")
                    print(f"   Summary Length: {len(analysis_local.summarized_findings)} chars")
                    
                except Exception as e:
                    print(f"   Error: {e}")
            else:
                print("🤖 Local AI Model: Not available")
            
            # Test External API
            print("\n🌐 External API:")
            try:
                rag_external = LocalAIEnhancedRAG(use_local_model=False)
                
                start_time = time.time()
                analysis_external = await rag_external.research_and_analyze(query)
                external_time = time.time() - start_time
                
                print(f"   Response Time: {external_time:.2f}s")
                print(f"   Papers Found: {analysis_external.processing_metadata['papers_found']}")
                print(f"   Model: {analysis_external.processing_metadata['llm_model_used']}")
                print(f"   Summary Length: {len(analysis_external.summarized_findings)} chars")
                
                # Compare performance
                if local_model_available:
                    print(f"\n📊 Performance Comparison:")
                    if local_time < external_time:
                        print(f"   ⚡ Local model is {external_time/local_time:.1f}x faster")
                    else:
                        print(f"   ⚡ External API is {local_time/external_time:.1f}x faster")
                
            except Exception as e:
                print(f"   Error: {e}")
        
        print(f"\n✅ Model comparison completed")
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")

async def performance_benchmark():
    """Run performance benchmark on local model"""
    print(f"\n⏱️  Performance Benchmark")
    print("=" * 60)
    
    try:
        from rag.local_ai_enhanced_rag import LocalAIEnhancedRAG
        
        # Find local model
        model_dir = Path("data/models/local_ai")
        if not model_dir.exists() or not list(model_dir.glob("*.pt")):
            print("❌ No local model found for benchmarking")
            return
        
        model_path = str(list(model_dir.glob("*.pt"))[0])
        rag = LocalAIEnhancedRAG(use_local_model=True, local_model_path=model_path)
        
        # Benchmark queries
        benchmark_queries = [
            "machine learning",
            "deep learning neural networks",
            "computer vision object detection",
            "natural language processing transformers",
            "reinforcement learning algorithms"
        ]
        
        print(f"Running {len(benchmark_queries)} benchmark queries...")
        
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(benchmark_queries, 1):
            print(f"\n{i}/{len(benchmark_queries)}: '{query}'")
            
            try:
                start_time = time.time()
                analysis = await rag.research_and_analyze(query)
                end_time = time.time()
                
                query_time = end_time - start_time
                total_time += query_time
                successful_queries += 1
                
                print(f"   Time: {query_time:.2f}s")
                print(f"   Papers: {analysis.processing_metadata['papers_found']}")
                print(f"   Status: ✅ Success")
                
            except Exception as e:
                print(f"   Status: ❌ Failed ({e})")
        
        # Calculate statistics
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"\n📊 Benchmark Results:")
            print(f"   Successful Queries: {successful_queries}/{len(benchmark_queries)}")
            print(f"   Average Response Time: {avg_time:.2f}s")
            print(f"   Total Time: {total_time:.2f}s")
            print(f"   Success Rate: {(successful_queries/len(benchmark_queries)*100):.1f}%")
            
            # Performance rating
            if avg_time < 2:
                rating = "Excellent"
            elif avg_time < 5:
                rating = "Good"
            elif avg_time < 10:
                rating = "Average"
            else:
                rating = "Needs Improvement"
            
            print(f"   Performance Rating: {rating}")
        else:
            print(f"\n❌ All benchmark queries failed")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

async def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Local AI Model")
    parser.add_argument("--model-path", type=str, help="Path to specific model to test")
    parser.add_argument("--standalone", action="store_true", help="Test model standalone")
    parser.add_argument("--integrated", action="store_true", help="Test integrated RAG system")
    parser.add_argument("--compare", action="store_true", help="Compare local vs external models")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Auto-discover model if not specified
    if not args.model_path:
        model_dir = Path("data/models/local_ai")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                args.model_path = str(model_files[0])
                print(f"🔍 Auto-discovered model: {Path(args.model_path).name}")
    
    # Run tests based on arguments
    if args.all or not any([args.standalone, args.integrated, args.compare, args.benchmark]):
        # Run all tests by default
        if args.model_path:
            await test_local_model_standalone(args.model_path)
        await test_integrated_rag_system()
        await compare_models()
        await performance_benchmark()
    else:
        if args.standalone and args.model_path:
            await test_local_model_standalone(args.model_path)
        
        if args.integrated:
            await test_integrated_rag_system()
        
        if args.compare:
            await compare_models()
        
        if args.benchmark:
            await performance_benchmark()
    
    print(f"\n🎉 Testing completed!")
    print(f"\n💡 Next steps:")
    print(f"   • Train a better model: python scripts/train_local_model.py --profile quality")
    print(f"   • Use in GUI: python launch_gui.py")
    print(f"   • Monitor performance: python scripts/monitor_performance.py")

if __name__ == "__main__":
    asyncio.run(main()) 