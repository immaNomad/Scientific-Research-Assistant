#!/usr/bin/env python3
"""
AI Research Assistant Setup Script
Automatically sets up the database, downloads papers, and trains models
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 AI Research Assistant Setup                            ║
║                                                                              ║
║  This script will set up your local research assistant with:                ║
║  • Research paper database (146+ papers)                                    ║
║  • Custom trained AI models                                                 ║
║  • Search optimization                                                       ║
║  • Configuration files                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate', 
        'loguru', 'requests', 'pandas', 'numpy', 'scikit-learn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install requirements first:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def create_directories():
    """Create required directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/papers",
        "data/models/local_ai",
        "data/cache",
        "data/embeddings", 
        "data/feedback",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def setup_config():
    """Setup configuration files"""
    print("⚙️  Setting up configuration...")
    
    config_source = Path("config/api_keys.example.py")
    config_target = Path("config/api_keys.py")
    
    if not config_target.exists():
        if config_source.exists():
            import shutil
            shutil.copy(config_source, config_target)
            print("   ✅ Configuration file created")
        else:
            print("   ⚠️  Warning: Example config not found")
    else:
        print("   ✅ Configuration file already exists")

async def populate_database():
    """Populate the database with research papers"""
    print("📚 Populating database with research papers...")
    print("   This may take 10-15 minutes...")
    
    try:
        # Add project root to Python path
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        # Import and run the population script
        from scripts.populate_database import main as populate_main
        await populate_main()
        
        print("   ✅ Database populated successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Database population failed: {e}")
        print("   You can run this manually later:")
        print("   python scripts/populate_database.py")
        return False

async def train_models():
    """Train the local AI models"""
    print("🤖 Training local AI models...")
    print("   This may take 20-30 minutes...")
    
    try:
        # Add project root to Python path
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        # Import and run the training script
        from scripts.train_local_model import main as train_main
        await train_main()
        
        print("   ✅ Models trained successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
        print("   You can run this manually later:")
        print("   python scripts/train_local_model.py")
        return False

def test_setup():
    """Test the setup"""
    print("🧪 Testing setup...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from database.paper_db import PaperDatabase
        from models.llm_client import LLMManager
        
        # Test database
        db = PaperDatabase()
        paper_count = len(db.get_all_papers())
        print(f"   ✅ Database: {paper_count} papers")
        
        # Test models
        llm = LLMManager()
        print("   ✅ Models: Loading successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Setup test failed: {e}")
        return False

async def main():
    """Main setup function"""
    print_banner()
    
    if not check_requirements():
        sys.exit(1)
    
    create_directories()
    setup_config()
    
    # Ask user about setup options
    print("\n🎯 Setup Options:")
    print("1. Full setup (recommended) - Database + Models")
    print("2. Database only")
    print("3. Models only")
    print("4. Skip data setup")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    success = True
    
    if choice in ["1", "2"]:
        success &= await populate_database()
    
    if choice in ["1", "3"]:
        success &= await train_models()
    
    if choice in ["1", "2", "3"]:
        success &= test_setup()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now run the application:")
        print("   python launch_gui.py")
        print("   python main.py")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Check the error messages above and run the failed commands manually.")

if __name__ == "__main__":
    asyncio.run(main()) 