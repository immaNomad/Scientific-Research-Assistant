#!/usr/bin/env python3
"""
Local AI Model Training Script
Train a custom AI model on your research database
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Configure logging for training"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "model_training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_banner():
    """Print training banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                        🤖 Local AI Model Training                             ║
║                                                                                ║
║  Train your own AI model using your local research paper database.            ║
║  No external APIs required - complete privacy and control.                    ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if training requirements are met"""
    print("\n🔍 Checking training requirements...")
    
    # Check PyTorch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ PyTorch available on device: {device}")
        
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️  Using CPU - training will be slower")
            
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ Transformers library: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found. Install with: pip install transformers")
        return False
    
    # Check database
    try:
        from database.paper_db import PaperDatabase
        db = PaperDatabase()
        stats = db.get_stats()
        
        if stats['total_papers'] == 0:
            print("❌ No papers in database. Run 'python scripts/populate_database.py' first")
            return False
        
        print(f"✅ Database ready: {stats['total_papers']} papers")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    return True

def estimate_training_time(num_papers: int, device: str, epochs: int = 3) -> str:
    """Estimate training time based on system specs"""
    if device == "cuda":
        # GPU training
        minutes_per_epoch = max(1, num_papers // 50)  # Rough estimate
    else:
        # CPU training
        minutes_per_epoch = max(5, num_papers // 10)  # Much slower on CPU
    
    total_minutes = minutes_per_epoch * epochs
    
    if total_minutes < 60:
        return f"{total_minutes} minutes"
    else:
        hours = total_minutes // 60
        remaining_minutes = total_minutes % 60
        return f"{hours}h {remaining_minutes}m"

async def train_model(config_name: str = "default", custom_config: dict = None):
    """Train the local AI model"""
    from models.local_ai_model import ModelConfig, LocalAITrainer, train_local_model
    from database.paper_db import PaperDatabase
    
    print("\n🚀 Starting model training...")
    
    # Setup configuration
    if custom_config:
        config = ModelConfig(**custom_config)
    elif config_name == "fast":
        config = ModelConfig(
            num_epochs=2,
            batch_size=16,
            max_papers_for_training=200,
            learning_rate=3e-5
        )
    elif config_name == "quality":
        config = ModelConfig(
            num_epochs=5,
            batch_size=4,
            max_papers_for_training=1000,
            learning_rate=1e-5
        )
    elif config_name == "cpu":
        config = ModelConfig(
            num_epochs=2,
            batch_size=2,
            max_papers_for_training=100,
            learning_rate=2e-5
        )
    else:  # default
        config = ModelConfig()
    
    # Show training configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max papers: {config.max_papers_for_training}")
    
    # Estimate training time
    db = PaperDatabase()
    num_papers = min(len(db.get_all_papers()), config.max_papers_for_training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimated_time = estimate_training_time(num_papers, device, config.num_epochs)
    
    print(f"\n⏱️  Estimated training time: {estimated_time}")
    print(f"   Training on {num_papers} papers")
    print(f"   Device: {device.upper()}")
    
    # Confirm training
    response = input("\nContinue with training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return None
    
    # Start training
    start_time = datetime.now()
    print(f"\n🏋️  Training started at {start_time.strftime('%H:%M:%S')}")
    
    try:
        # Train model
        model_path = train_local_model(config=config)
        
        # Training completed
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Training completed!")
        print(f"   Duration: {duration}")
        print(f"   Model saved: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training error: {e}")
        return None

async def test_model(model_path: str):
    """Test the trained model"""
    from models.local_ai_model import LocalAIInference
    from database.paper_db import PaperDatabase
    
    print(f"\n🧪 Testing trained model...")
    
    try:
        # Load model
        model = LocalAIInference(model_path)
        
        # Get test data
        db = PaperDatabase()
        papers = db.get_all_papers()[:5]  # Test with first 5 papers
        
        test_queries = [
            "machine learning",
            "deep learning",
            "neural networks",
            "computer vision",
            "artificial intelligence"
        ]
        
        print(f"\n📊 Model Performance Test:")
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            # Test relevance scoring
            if papers:
                relevance = model.calculate_relevance(query, papers[0].abstract)
                print(f"   Relevance score: {relevance:.2f}/5.0")
            
            # Test summary generation
            context = papers[0].abstract if papers else "Sample research context"
            summary = model.generate_summary(query, context)
            print(f"   Summary: {summary[:100]}...")
        
        print(f"\n✅ Model testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Model testing failed: {e}")
        logging.error(f"Testing error: {e}")

def interactive_training():
    """Interactive training mode"""
    print("\n🔧 Interactive Training Configuration")
    
    # Get user preferences
    print("\nChoose training profile:")
    print("1. Fast - Quick training for testing (2 epochs, 200 papers)")
    print("2. Balanced - Good quality and speed (3 epochs, 500 papers)")
    print("3. Quality - Best results (5 epochs, 1000 papers)")
    print("4. CPU Optimized - For systems without GPU (2 epochs, 100 papers)")
    print("5. Custom - Specify your own parameters")
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            return "fast"
        elif choice == "2":
            return "default"
        elif choice == "3":
            return "quality"
        elif choice == "4":
            return "cpu"
        elif choice == "5":
            return configure_custom_training()
        else:
            print("Invalid choice. Please enter 1-5.")

def configure_custom_training():
    """Configure custom training parameters"""
    print("\n⚙️  Custom Training Configuration")
    
    try:
        epochs = int(input("Number of epochs (1-10) [3]: ") or "3")
        batch_size = int(input("Batch size (1-32) [8]: ") or "8")
        max_papers = int(input("Max papers for training (10-1000) [500]: ") or "500")
        learning_rate = float(input("Learning rate (1e-6 to 1e-4) [2e-5]: ") or "2e-5")
        
        return {
            'num_epochs': max(1, min(10, epochs)),
            'batch_size': max(1, min(32, batch_size)),
            'max_papers_for_training': max(10, min(1000, max_papers)),
            'learning_rate': max(1e-6, min(1e-4, learning_rate))
        }
        
    except ValueError:
        print("Invalid input. Using default configuration.")
        return "default"

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Local AI Model")
    parser.add_argument("--profile", choices=["fast", "balanced", "quality", "cpu", "custom"],
                       default="balanced", help="Training profile")
    parser.add_argument("--interactive", action="store_true", help="Interactive configuration")
    parser.add_argument("--test-only", type=str, help="Test existing model (provide model path)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--max-papers", type=int, help="Maximum papers for training")
    
    args = parser.parse_args()
    
    setup_logging()
    print_banner()
    
    # Test existing model only
    if args.test_only:
        if os.path.exists(args.test_only):
            await test_model(args.test_only)
        else:
            print(f"❌ Model file not found: {args.test_only}")
        return
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing dependencies.")
        return
    
    # Configure training
    if args.interactive:
        config = interactive_training()
    else:
        config = args.profile
        
        # Override with command line args
        if any([args.epochs, args.batch_size, args.max_papers]):
            custom_config = {}
            if args.epochs:
                custom_config['num_epochs'] = args.epochs
            if args.batch_size:
                custom_config['batch_size'] = args.batch_size
            if args.max_papers:
                custom_config['max_papers_for_training'] = args.max_papers
            config = custom_config
    
    # Train model
    if isinstance(config, dict):
        model_path = await train_model(custom_config=config)
    else:
        model_path = await train_model(config_name=config)
    
    # Test trained model
    if model_path:
        await test_model(model_path)
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Test your model: python scripts/train_local_model.py --test-only {model_path}")
        print(f"   2. Integrate with research assistant (see integration guide)")
        print(f"   3. Use model in local RAG pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 