"""
Minimal TestCraft UI Demo - Clean and Stylish

This demo showcases the redesigned minimalist UI system with:
- Clean, subtle color palette
- Simplified status indicators  
- Minimal progress visualization
- Essential information focus
- Reduced visual clutter
- Stylish but unobtrusive design
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import random
import sys

# Add the parent directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter
from testcraft.adapters.io.file_status_tracker import (
    FileStatusTracker, 
    FileStatus, 
    LiveFileTracking
)


async def demo_minimal_file_tracking():
    """Demonstrate the clean, minimal live file tracking."""
    ui = EnhancedUIAdapter()
    
    ui.console.print("\n[header]minimal file tracking[/]\n")
    
    # Clean sample files list
    sample_files = [
        "src/auth.py",
        "src/models.py", 
        "src/utils.py",
        "src/api.py",
        "src/config.py"
    ]
    
    ui.console.print(f"processing [accent]{len(sample_files)}[/] files\n")
    
    with LiveFileTracking(ui, "testcraft") as live_tracker:
        file_tracker = live_tracker.initialize_and_start(sample_files)
        
        # Simulate clean processing
        for file_path in sample_files:
            # Analysis phase
            file_tracker.update_file_status(
                file_path,
                FileStatus.ANALYZING,
                operation="analyze",
                step="parsing source",
                progress=25.0
            )
            await asyncio.sleep(0.4)
            
            # Generation phase  
            file_tracker.update_file_status(
                file_path,
                FileStatus.GENERATING,
                operation="generate", 
                step="creating tests",
                progress=50.0
            )
            await asyncio.sleep(0.8)
            
            # Writing phase
            file_tracker.update_file_status(
                file_path,
                FileStatus.WRITING,
                operation="write",
                step="saving file",
                progress=75.0
            )
            await asyncio.sleep(0.2)
            
            # Completion
            success = random.random() > 0.2  # 80% success
            if success:
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.COMPLETED,
                    operation="complete",
                    step="done",
                    progress=100.0
                )
                file_tracker.update_generation_result(
                    file_path, 
                    True, 
                    random.randint(3, 8)
                )
            else:
                file_tracker.update_file_status(
                    file_path,
                    FileStatus.FAILED,
                    operation="failed",
                    step="generation error", 
                    progress=0.0
                )
        
        await asyncio.sleep(2)
        stats = file_tracker.get_summary_stats()
    
    # Clean summary
    ui.console.print(f"\ncompleted [success]{stats['completed']}[/] of [primary]{stats['total_files']}[/] files")
    ui.console.print(f"generated [accent]{stats['total_tests_generated']}[/] tests in [muted]{stats['total_duration']:.1f}s[/]\n")


def demo_minimal_tables():
    """Demo clean, minimal table design."""
    ui = EnhancedUIAdapter()
    
    ui.console.print("[header]minimal tables[/]\n")
    
    # Sample file data with clean structure
    files_data = [
        {
            "file_path": "src/main.py",
            "status": "completed", 
            "progress": 100.0,
            "tests_generated": 5,
            "duration": 2.3
        },
        {
            "file_path": "src/auth.py",
            "status": "active",
            "progress": 60.0, 
            "tests_generated": 0,
            "duration": 1.1
        },
        {
            "file_path": "src/models.py",
            "status": "failed",
            "progress": 30.0,
            "tests_generated": 0,
            "duration": 0.8
        },
        {
            "file_path": "src/utils.py",
            "status": "waiting",
            "progress": 0.0,
            "tests_generated": 0,
            "duration": 0.0
        }
    ]
    
    ui.display_file_progress_table(files_data, "processing status")


def demo_minimal_success():
    """Demo minimal success display."""
    ui = EnhancedUIAdapter()
    
    ui.console.print("[header]minimal success display[/]\n")
    
    # Clean success data
    summary_data = {
        "message": "generated tests for 4 files",
        "metrics": {
            "generation": {
                "duration": 12.3,
                "items_processed": 4,
                "success_rate": 0.75
            }
        }
    }
    
    ui.display_success_summary(summary_data)


def demo_minimal_errors():
    """Demo minimal error handling."""
    ui = EnhancedUIAdapter()
    
    ui.console.print("[header]minimal error handling[/]\n")
    
    # Clean error examples
    ui.display_error_with_suggestions(
        "invalid config file format",
        [
            "check toml syntax",
            "verify required fields", 
            "run testcraft init-config"
        ],
        "config error"
    )


async def main():
    """Run the minimal UI demonstration."""
    ui = EnhancedUIAdapter()
    
    # Clean welcome
    ui.console.clear()
    ui.console.print("[header]testcraft minimal ui[/]\n")
    ui.console.print("clean • minimal • stylish\n")
    ui.console.print("[muted]press enter to continue...[/]")
    input()
    
    try:
        # Demo 1: Minimal file tracking
        await demo_minimal_file_tracking()
        
        ui.console.print("[muted]press enter to continue...[/]")
        input()
        
        # Demo 2: Minimal tables
        demo_minimal_tables()
        
        ui.console.print("\n[muted]press enter to continue...[/]")
        input()
        
        # Demo 3: Minimal success
        demo_minimal_success()
        
        ui.console.print("\n[muted]press enter to continue...[/]")
        input()
        
        # Demo 4: Minimal errors
        demo_minimal_errors()
        
        # Clean finish
        ui.console.print("\n[header]demo complete[/]")
        ui.console.print("\n[success]✓[/] clean design")
        ui.console.print("[success]✓[/] essential information")
        ui.console.print("[success]✓[/] minimal visual noise")
        ui.console.print("[success]✓[/] stylish but subtle\n")
        
    except KeyboardInterrupt:
        ui.console.print("\n[muted]demo interrupted[/]")
    except Exception as e:
        ui.display_error(f"demo failed: {str(e)}", "error")
    finally:
        ui.finalize()


if __name__ == "__main__":
    asyncio.run(main())
