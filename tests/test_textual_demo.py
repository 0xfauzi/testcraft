"""
Basic tests for TestCraft Textual TUI components.

These tests demonstrate how to test Textual applications using
the App.run_test() method and Pilot for interaction simulation.
"""

import pytest
from pathlib import Path
import sys

# Add testcraft to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from testcraft.adapters.textual.app import TestCraftTextualApp
from testcraft.adapters.textual.widgets import FileTable, StatsPanel
from testcraft.adapters.textual.events import FileStatusChanged, StatsUpdated


class TestTextualApp:
    """Test the main TestCraft Textual application."""
    
    def test_app_creation(self):
        """Test that the app can be created successfully."""
        app = TestCraftTextualApp()
        assert app is not None
        assert app.TITLE == "TestCraft"
        assert app.SUB_TITLE == "AI-Powered Test Generation"
    
    def test_app_keybindings(self):
        """Test that the app has the expected keybindings."""
        app = TestCraftTextualApp()
        
        # Check that key bindings are defined (BINDINGS is a list of tuples)
        binding_keys = [binding[0] for binding in app.BINDINGS]  # First element is the key
        
        expected_keys = ["ctrl+c,q", "g", "a", "c", "s", "w", "/", "?", "d", "r", "l"]
        
        for key in expected_keys:
            assert key in binding_keys
    
    @pytest.mark.asyncio
    async def test_app_basic_functionality(self):
        """Test basic app functionality using Textual's testing framework."""
        app = TestCraftTextualApp()
        
        # Use Textual's testing capabilities
        async with app.run_test() as pilot:
            # Test initial state
            assert app.current_operation is None
            
            # Test file status update
            app.update_file_status("test.py", "running", 50.0, 2, 1.5)
            assert "test.py" in app.file_states
            assert app.file_states["test.py"]["status"] == "running"
            
            # Test stats update
            stats = {
                "files_total": 5,
                "files_done": 2,
                "tests_generated": 10,
            }
            app.update_stats(stats)
            assert app.operation_stats["files_total"] == 5


class TestFileTable:
    """Test the FileTable widget logic without app context."""
    
    def test_file_table_constants(self):
        """Test that FileTable constants are properly defined."""
        assert FileTable.COLUMNS is not None
        assert len(FileTable.COLUMNS) > 0
        assert FileTable.STATUS_STYLES is not None
    
    def test_file_table_data_operations(self):
        """Test file table data operations without app context."""
        table = FileTable()
        
        # Test internal data storage
        table._file_data["example.py"] = {
            "file": "example.py",
            "status": "running",
            "progress": 75.0,
            "tests": 3,
            "duration": 2.5,
            "error": None,
        }
        
        # Test filtering
        all_data = table._get_filtered_data()
        assert len(all_data) == 1
        
        # Test data formatting
        file_data = table._file_data["example.py"]
        formatted_cells = table._format_row_cells(file_data)
        assert len(formatted_cells) == len(table.COLUMNS)
        assert "example.py" in formatted_cells[0]  # File name
        assert "Running" in formatted_cells[1]     # Status (capitalized)
    
    def test_file_table_sorting_logic(self):
        """Test file table sorting logic."""
        table = FileTable()
        
        # Add test data
        test_data = [
            ("a.py", {"file": "a.py", "status": "done", "tests": 3, "duration": 1.0}),
            ("z.py", {"file": "z.py", "status": "done", "tests": 5, "duration": 2.0}),
            ("m.py", {"file": "m.py", "status": "done", "tests": 1, "duration": 3.0}),
        ]
        
        # Test sorting by file name (default)
        table.sort_column = "file"
        sorted_data = table._get_sorted_data(test_data)
        file_names = [item[0] for item in sorted_data]
        assert file_names == ["a.py", "m.py", "z.py"]
        
        # Test sorting by tests (numeric)
        table.sort_column = "tests"
        sorted_data = table._get_sorted_data(test_data)
        tests_counts = [item[1]["tests"] for item in sorted_data]
        assert tests_counts == [1, 3, 5]


class TestStatsPanel:
    """Test the StatsPanel widget."""
    
    def test_stats_panel_creation(self):
        """Test that StatsPanel can be created."""
        panel = StatsPanel()
        assert panel is not None
        assert panel._default_stats is not None
    
    def test_stats_panel_update(self):
        """Test updating statistics."""
        panel = StatsPanel()
        
        new_stats = {
            "files_total": 10,
            "files_done": 7,
            "files_failed": 1,
            "tests_generated": 35,
            "total_duration": 45.5,
        }
        
        panel.update_stats(new_stats)
        
        # Check that stats were updated
        assert panel.stats_data["files_total"] == 10
        assert panel.stats_data["files_done"] == 7
        assert panel.stats_data["files_failed"] == 1
        assert panel.stats_data["tests_generated"] == 35
        assert panel.stats_data["total_duration"] == 45.5
    
    def test_stats_calculation(self):
        """Test derived statistics calculation."""
        panel = StatsPanel()
        
        stats = {
            "files_total": 10,
            "files_done": 8,
            "files_failed": 2,
            "total_duration": 20.0,
        }
        
        calculated = panel._calculate_derived_stats(stats)
        
        # Check success rate calculation
        expected_success_rate = (8 / (8 + 2)) * 100  # 80%
        assert calculated["success_rate"] == expected_success_rate
        
        # Check average duration calculation
        expected_avg_duration = 20.0 / 8  # 2.5s
        assert calculated["avg_duration"] == expected_avg_duration
    
    def test_stats_reset(self):
        """Test resetting statistics."""
        panel = StatsPanel()
        
        # Update with some data
        panel.update_stats({"files_total": 10, "files_done": 5})
        assert panel.stats_data["files_total"] == 10
        
        # Reset
        panel.reset_stats()
        assert panel.stats_data["files_total"] == 0
        assert panel.stats_data["files_done"] == 0


class TestEvents:
    """Test custom Textual events."""
    
    def test_file_status_changed_event(self):
        """Test FileStatusChanged event creation."""
        event = FileStatusChanged(
            file_path="test.py",
            status="running", 
            progress=75.0,
            tests_generated=3,
            duration=2.5,
            error=None
        )
        
        assert event.file_path == "test.py"
        assert event.status == "running"
        assert event.progress == 75.0
        assert event.tests_generated == 3
        assert event.duration == 2.5
        assert event.error is None
    
    def test_stats_updated_event(self):
        """Test StatsUpdated event creation."""
        stats_data = {
            "files_total": 10,
            "files_done": 7,
            "tests_generated": 35,
        }
        
        event = StatsUpdated(stats_data)
        
        assert event.stats == stats_data
        assert event.stats["files_total"] == 10
        assert event.stats["files_done"] == 7
        assert event.stats["tests_generated"] == 35


# Integration test example
@pytest.mark.asyncio 
async def test_app_file_processing_integration():
    """Integration test showing file processing workflow."""
    app = TestCraftTextualApp()
    
    async with app.run_test() as pilot:
        # Simulate file processing workflow
        files_to_process = ["file1.py", "file2.py", "file3.py"]
        
        # Start processing
        for i, file_path in enumerate(files_to_process):
            app.update_file_status(file_path, "running", 0.0)
            
            # Simulate progress updates
            for progress in [25.0, 50.0, 75.0, 100.0]:
                tests_generated = int(progress / 25)  # Simple calculation
                app.update_file_status(
                    file_path, 
                    "running" if progress < 100 else "done",
                    progress,
                    tests_generated,
                    progress / 50  # Simple duration calculation
                )
        
        # Update final stats
        app.update_stats({
            "files_total": len(files_to_process),
            "files_done": len(files_to_process),
            "files_failed": 0,
            "tests_generated": 12,  # 4 tests per file
        })
        
        # Verify final state
        assert len(app.file_states) == 3
        for file_path in files_to_process:
            assert app.file_states[file_path]["status"] == "done"
            assert app.file_states[file_path]["progress"] == 100.0
        
        assert app.operation_stats["files_total"] == 3
        assert app.operation_stats["files_done"] == 3
        assert app.operation_stats["tests_generated"] == 12


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
