#!/usr/bin/env python3
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
