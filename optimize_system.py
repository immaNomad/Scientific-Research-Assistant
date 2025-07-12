#!/usr/bin/env python3
"""
Research Assistant System Optimizer
Complete setup and optimization script for the research assistant
"""

import asyncio
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def install_requirements():
    """Install additional optimization requirements"""
    requirements = [
        "psutil",           # System monitoring
        "PyYAML",           # Configuration files
        "tqdm",             # Progress bars
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")

def print_banner():
    """Print optimization banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    🚀 Research Assistant System Optimizer                      ║
║                                                                                ║
║  This script will optimize your research assistant for better performance,    ║
║  populate the database with papers, and set up monitoring.                    ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 2 * 1024 * 1024 * 1024:  # Less than 2GB
            print("⚠️  Warning: Less than 2GB RAM available")
        else:
            print(f"✅ Available RAM: {memory.total / (1024**3):.1f} GB")
    except ImportError:
        print("⚠️  psutil not available - installing...")
        install_requirements()
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1:
            print("⚠️  Warning: Less than 1GB disk space available")
        else:
            print(f"✅ Available disk space: {free_gb:.1f} GB")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    return True

async def optimize_system():
    """Main optimization function"""
    print_banner()
    
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False
    
    # Install requirements
    print("\n📦 Installing optimization requirements...")
    install_requirements()
    
    # Import after installation
    try:
        from database.paper_db import PaperDatabase
        from database.enhanced_search import EnhancedSearchEngine
        from config.optimization_config import ConfigManager
        from analytics.performance_monitor import PerformanceMonitor
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Step 1: Setup configuration
    print("\n🔧 Setting up optimization configuration...")
    config_manager = ConfigManager()
    
    # Ask user for performance profile
    print("\nChoose a performance profile:")
    print("1. Fast - Optimized for speed")
    print("2. Balanced - Good balance of speed and quality")
    print("3. Quality - Optimized for research quality")
    print("4. Memory Optimized - For systems with limited RAM")
    
    while True:
        try:
            choice = input("Enter choice (1-4) [default: 2]: ").strip()
            if not choice:
                choice = "2"
            
            profile_map = {
                "1": "fast",
                "2": "balanced", 
                "3": "quality",
                "4": "memory_optimized"
            }
            
            if choice in profile_map:
                profile = profile_map[choice]
                config_manager.apply_performance_profile(profile)
                print(f"✅ Applied {profile} performance profile")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n❌ Optimization cancelled")
            return False
    
    # Step 2: Setup database
    print("\n🗄️  Setting up database...")
    db = PaperDatabase()
    stats = db.get_stats()
    
    if stats['total_papers'] == 0:
        print("📚 Database is empty. Populating with AI/ML papers...")
        
        # Import and run population script
        from database.paper_collector import collect_papers
        
        try:
            paper_ids = await collect_papers(50)  # Collect 50 papers
            print(f"✅ Successfully collected {len(paper_ids)} papers")
        except Exception as e:
            print(f"❌ Error collecting papers: {e}")
            print("⚠️  You can run 'python scripts/populate_database.py' later")
    else:
        print(f"✅ Database already has {stats['total_papers']} papers")
    
    # Step 3: Setup enhanced search
    print("\n🔍 Setting up enhanced search engine...")
    try:
        search_engine = EnhancedSearchEngine()
        
        # Test search
        test_results = await search_engine.search_literature("machine learning", max_results_per_source=5)
        print(f"✅ Search engine working - found {len(test_results)} test results")
        
        # Show search statistics
        index_stats = search_engine.get_index_stats()
        print(f"   📊 Indexed {index_stats['total_papers_indexed']} papers")
        print(f"   📝 {index_stats['keyword_index_size']} keywords indexed")
        
    except Exception as e:
        print(f"❌ Error setting up search engine: {e}")
    
    # Step 4: Setup performance monitoring
    print("\n📊 Setting up performance monitoring...")
    try:
        performance_monitor = PerformanceMonitor()
        
        # Test monitoring
        performance_monitor.log_query(
            "test query", 1.0, 5, "enhanced", "general", True
        )
        
        summary = performance_monitor.get_performance_summary()
        print(f"✅ Performance monitoring active - status: {summary['status']}")
        
    except Exception as e:
        print(f"❌ Error setting up monitoring: {e}")
    
    # Step 5: Generate optimization report
    print("\n📋 Generating optimization report...")
    
    recommendations = config_manager.get_optimization_recommendations()
    if recommendations:
        print("\n💡 Optimization recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("✅ No additional optimizations needed")
    
    # Step 6: Setup scripts
    print("\n📄 Setting up helper scripts...")
    
    # Create optimization shortcuts
    scripts_to_create = [
        ("quick_search.py", create_quick_search_script()),
        ("monitor_performance.py", create_monitor_script()),
        ("backup_database.py", create_backup_script())
    ]
    
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    for script_name, script_content in scripts_to_create:
        script_path = scripts_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)  # Make executable
        print(f"✅ Created {script_name}")
    
    # Final summary
    print("\n🎉 System optimization completed!")
    print("\n📊 Final Status:")
    print(f"   Database: {db.get_stats()['total_papers']} papers")
    print(f"   Configuration: {profile} profile")
    print(f"   Search: Enhanced engine with caching")
    print(f"   Monitoring: Performance tracking enabled")
    
    print("\n🚀 Quick Start Commands:")
    print("   python launch_gui.py                    # Launch GUI")
    print("   python scripts/quick_search.py          # Quick search test")
    print("   python scripts/monitor_performance.py   # View performance")
    print("   python scripts/populate_database.py     # Add more papers")
    
    return True

def create_quick_search_script():
    """Create a quick search test script"""
    return '''#!/usr/bin/env python3
"""Quick search test script"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def main():
    from database.enhanced_search import EnhancedSearchEngine
    
    search_engine = EnhancedSearchEngine()
    
    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "computer vision",
        "natural language processing"
    ]
    
    print("🔍 Quick Search Test")
    print("=" * 30)
    
    for query in queries:
        print(f"\\nSearching: {query}")
        try:
            results = await search_engine.search_literature(query, max_results_per_source=3)
            print(f"  Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result.title[:50]}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show cache stats
    cache_stats = search_engine.get_cache_stats()
    print(f"\\n📊 Cache Statistics:")
    print(f"   Size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['cache_hits']}/{cache_stats['cache_size']}")

if __name__ == "__main__":
    asyncio.run(main())
'''

def create_monitor_script():
    """Create a performance monitoring script"""
    return '''#!/usr/bin/env python3
"""Performance monitoring script"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    from analytics.performance_monitor import get_performance_monitor
    
    monitor = get_performance_monitor()
    
    print("📊 Performance Monitor")
    print("=" * 40)
    
    # Show performance summary
    summary = monitor.get_performance_summary()
    print(f"Status: {summary['status'].upper()}")
    
    if summary['warnings']:
        print("\\n⚠️  Warnings:")
        for warning in summary['warnings']:
            print(f"   • {warning}")
    
    # Show usage analytics
    analytics = monitor.get_usage_analytics(24)  # Last 24 hours
    print(f"\\n📈 Usage Analytics (24h):")
    print(f"   Average response time: {analytics.average_response_time:.2f}s")
    print(f"   Success rate: {analytics.search_success_rate:.1f}%")
    
    if analytics.most_common_queries:
        print(f"\\n🔥 Most Common Queries:")
        for query, count in analytics.most_common_queries[:5]:
            print(f"   • {query} ({count} times)")
    
    # Generate full report
    print("\\n" + "=" * 40)
    print(monitor.generate_report(24))

if __name__ == "__main__":
    main()
'''

def create_backup_script():
    """Create a database backup script"""
    return '''#!/usr/bin/env python3
"""Database backup script"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

def main():
    # Backup database
    db_path = Path("data/papers/papers.db")
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    # Create backup directory
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"papers_backup_{timestamp}.db"
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backed up to {backup_path}")
        
        # Show backup size
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        print(f"   Backup size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
    
    # Clean old backups (keep last 5)
    backups = sorted(backup_dir.glob("papers_backup_*.db"), reverse=True)
    if len(backups) > 5:
        for old_backup in backups[5:]:
            old_backup.unlink()
            print(f"   Cleaned old backup: {old_backup.name}")

if __name__ == "__main__":
    main()
'''

async def interactive_optimization():
    """Interactive optimization mode"""
    print("🔧 Interactive Optimization Mode")
    print("Available commands:")
    print("  optimize - Run full optimization")
    print("  config - Show current configuration")
    print("  profile <name> - Apply performance profile")
    print("  test - Run system tests")
    print("  status - Show system status")
    print("  exit - Exit interactive mode")
    
    while True:
        try:
            command = input("\noptimize> ").strip().lower()
            
            if command == "exit":
                break
            elif command == "optimize":
                await optimize_system()
            elif command == "config":
                from config.optimization_config import get_config_manager
                config = get_config_manager()
                print(config.export_config())
            elif command.startswith("profile "):
                profile_name = command[8:]
                from config.optimization_config import get_config_manager
                config = get_config_manager()
                try:
                    config.apply_performance_profile(profile_name)
                    print(f"✅ Applied {profile_name} profile")
                except ValueError as e:
                    print(f"❌ {e}")
            elif command == "test":
                await run_system_tests()
            elif command == "status":
                await show_system_status()
            else:
                print("Unknown command. Type 'exit' to quit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

async def run_system_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    # Test database
    try:
        from database.paper_db import PaperDatabase
        db = PaperDatabase()
        stats = db.get_stats()
        print(f"✅ Database: {stats['total_papers']} papers")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    # Test search
    try:
        from database.enhanced_search import EnhancedSearchEngine
        search = EnhancedSearchEngine()
        results = await search.search_literature("test", max_results_per_source=1)
        print(f"✅ Search: {len(results)} results")
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    # Test monitoring
    try:
        from analytics.performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()
        summary = monitor.get_performance_summary()
        print(f"✅ Monitoring: {summary['status']}")
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")

async def show_system_status():
    """Show current system status"""
    print("📊 System Status")
    print("=" * 30)
    
    # Database status
    try:
        from database.paper_db import PaperDatabase
        db = PaperDatabase()
        stats = db.get_stats()
        print(f"Database: {stats['total_papers']} papers, {stats['papers_with_pdf']} PDFs")
    except Exception as e:
        print(f"Database: Error - {e}")
    
    # Search engine status
    try:
        from database.enhanced_search import EnhancedSearchEngine
        search = EnhancedSearchEngine()
        index_stats = search.get_index_stats()
        cache_stats = search.get_cache_stats()
        print(f"Search: {index_stats['total_papers_indexed']} indexed, {cache_stats['cache_size']} cached")
    except Exception as e:
        print(f"Search: Error - {e}")
    
    # Performance status
    try:
        from analytics.performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()
        summary = monitor.get_performance_summary()
        print(f"Performance: {summary['status']}")
    except Exception as e:
        print(f"Performance: Error - {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Research Assistant System Optimizer")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--profile", choices=["fast", "balanced", "quality", "memory_optimized"],
                       help="Apply specific performance profile")
    parser.add_argument("--test", action="store_true", help="Run system tests only")
    parser.add_argument("--status", action="store_true", help="Show system status only")
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_optimization())
    elif args.profile:
        from config.optimization_config import get_config_manager
        config = get_config_manager()
        config.apply_performance_profile(args.profile)
        print(f"✅ Applied {args.profile} profile")
    elif args.test:
        asyncio.run(run_system_tests())
    elif args.status:
        asyncio.run(show_system_status())
    else:
        asyncio.run(optimize_system())

if __name__ == "__main__":
    main() 