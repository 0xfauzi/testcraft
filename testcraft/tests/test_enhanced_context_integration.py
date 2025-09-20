"""
Test the enhanced context integration system.

Tests the complete pipeline from packaging detection through validation
to ensure the weather scheduler issues are resolved.
"""

import tempfile
from pathlib import Path

from ..application.generation.services.enhanced_context_builder import (
    EnrichedContextBuilder,
)
from ..application.generation.services.generator_guardrails import (
    TestContentValidator,
)
from ..application.generation.services.packaging_detector import (
    EntityInterfaceDetector,
    PackagingDetector,
)


class TestPackagingDetection:
    """Test packaging detection for various project layouts."""

    def test_src_source_root_detection(self):
        """Test that src/ without __init__.py is detected as source root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create src/ layout without __init__.py (source root, not package)
            src_dir = project_root / "src"
            src_dir.mkdir()

            # Create a package inside src/
            weather_pkg = src_dir / "weather_collector"
            weather_pkg.mkdir()
            (weather_pkg / "__init__.py").write_text("")

            # Create the scheduler module
            scheduler_file = weather_pkg / "scheduler.py"
            scheduler_file.write_text("""
class WeatherScheduler:
    def __init__(self):
        pass
""")

            # Create pyproject.toml
            (project_root / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools"]

[tool.setuptools]
package-dir = {"" = "src"}
""")

            # Test packaging detection
            packaging_info = PackagingDetector.detect_packaging(project_root)

            assert not packaging_info.src_is_package
            assert "src." in packaging_info.disallowed_import_prefixes

            # Test import path resolution
            canonical_import = packaging_info.get_canonical_import(scheduler_file)
            assert canonical_import == "weather_collector.scheduler"

            # Verify disallowed imports
            assert not packaging_info.is_import_allowed(
                "src.weather_collector.scheduler"
            )
            assert packaging_info.is_import_allowed("weather_collector.scheduler")

    def test_flat_layout_detection(self):
        """Test flat project layout detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create flat layout
            weather_pkg = project_root / "weather_collector"
            weather_pkg.mkdir()
            (weather_pkg / "__init__.py").write_text("")

            scheduler_file = weather_pkg / "scheduler.py"
            scheduler_file.write_text("class WeatherScheduler: pass")

            packaging_info = PackagingDetector.detect_packaging(project_root)

            # Should detect project root as source root
            assert project_root in packaging_info.source_roots

            canonical_import = packaging_info.get_canonical_import(scheduler_file)
            assert canonical_import == "weather_collector.scheduler"


class TestEntityDetection:
    """Test entity interface detection for ORM models."""

    def test_sqlalchemy_model_detection(self):
        """Test detection of SQLAlchemy models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_file = Path(temp_dir) / "models.py"
            models_file.write_text("""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    country = Column(String(100))

    def __init__(self, name, country, latitude, longitude, timezone_offset=0):
        self.name = name
        self.country = country

class WeatherReading(Base):
    __tablename__ = 'weather_readings'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer)
    temperature = Column(Integer)
""")

            entity_info = EntityInterfaceDetector.detect_entities(models_file)

            assert entity_info["has_orm_models"]
            assert "City" in entity_info["entities"]
            assert "WeatherReading" in entity_info["entities"]

            city_info = entity_info["entities"]["City"]
            assert city_info["kind"] == "sqlalchemy.Model"
            assert not city_info["instantiate_real"]
            assert "name" in city_info["attributes_read_by_uut"]
            assert "country" in city_info["attributes_read_by_uut"]

    def test_regular_class_detection(self):
        """Test detection of regular classes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service_file = Path(temp_dir) / "service.py"
            service_file.write_text("""
class WeatherAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_current_weather(self, city):
        pass
""")

            entity_info = EntityInterfaceDetector.detect_entities(service_file)

            assert not entity_info["has_orm_models"]
            assert "WeatherAPIClient" in entity_info["entities"]

            client_info = entity_info["entities"]["WeatherAPIClient"]
            assert client_info["kind"] == "regular_class"
            assert client_info["instantiate_real"]


class TestEnrichedContextBuilder:
    """Test the enriched context builder."""

    def test_weather_scheduler_context(self):
        """Test context building for weather scheduler scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Set up project structure
            src_dir = project_root / "src"
            src_dir.mkdir()

            weather_pkg = src_dir / "weather_collector"
            weather_pkg.mkdir()
            (weather_pkg / "__init__.py").write_text("")

            # Create scheduler file
            scheduler_file = weather_pkg / "scheduler.py"
            scheduler_file.write_text("""
import logging
import time
from datetime import datetime
from typing import List, Optional
import schedule
from rich.console import Console
from rich.table import Table
from .api_client import WeatherAPIClient
from .database import get_db
from .models import City, WeatherReading

console = Console()
logger = logging.getLogger(__name__)

class WeatherScheduler:
    def __init__(self):
        self.api_client = WeatherAPIClient()
        self.running = False

    def collect_weather_data(self, cities: List[str]) -> None:
        console.print(f"Collecting weather data at {datetime.now()}")

        with get_db() as db:
            for city_name in cities:
                try:
                    data = self.api_client.get_current_weather(city_name)
                    # ... rest of implementation
                except Exception as e:
                    logger.error(f"Failed to collect data for {city_name}: {e}")

            db.commit()
""")

            # Create models file with ORM models
            models_file = weather_pkg / "models.py"
            models_file.write_text("""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))

class WeatherReading(Base):
    __tablename__ = 'weather_readings'
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer)
""")

            # Create pyproject.toml
            (project_root / "pyproject.toml").write_text("""
[tool.setuptools]
package-dir = {"" = "src"}
""")

            # Build enriched context
            builder = EnrichedContextBuilder()
            enriched_context = builder.build_enriched_context(
                source_file=scheduler_file,
                project_root=project_root,
            )

            # Verify packaging information
            packaging = enriched_context["packaging"]
            assert not packaging["src_is_package"]
            assert "src." in packaging["disallowed_import_prefixes"]

            # Verify import information
            imports = enriched_context["imports"]
            assert imports["module_path"] == "weather_collector.scheduler"
            assert "weather_collector.scheduler" in imports["canonical_import"]
            assert imports["validation_status"] == "validated"

            # Verify entity detection
            entities = enriched_context["entities"]
            assert "WeatherScheduler" in entities

            # Verify boundaries detection
            boundaries = enriched_context["boundaries_to_mock"]
            assert "time" in boundaries
            assert "database" in boundaries
            assert "time.sleep" in boundaries["time"]
            assert "get_db" in boundaries["database"]

            # Verify safety rules
            safety_rules = enriched_context["test_safety_rules"]
            rule_text = " ".join(safety_rules)
            assert "Never use import prefixes: 'src.'" in rule_text
            assert (
                "Never create domain objects inside @pytest.mark.parametrize"
                in rule_text
            )

            # Test LLM formatting
            formatted = builder.format_for_llm(enriched_context)
            assert (
                "import_statement: from weather_collector.scheduler import WeatherScheduler"
                in formatted
            )
            assert "disallowed_prefixes: ['src.']" in formatted
            assert "Side-Effect Boundaries to Mock" in formatted
            assert "Test Safety Rules" in formatted


class TestGeneratorGuardrails:
    """Test the generator guardrails system."""

    def test_import_validation(self):
        """Test validation of import statements."""
        enriched_context = {
            "packaging": {
                "disallowed_import_prefixes": ["src."],
            },
            "entities": {},
            "test_safety_rules": [],
        }

        # Test bad import
        bad_test_content = """
import pytest
import src.weather_collector.scheduler as scheduler_module

def test_weather_scheduler():
    ws = scheduler_module.WeatherScheduler()
    assert ws is not None
"""

        is_valid, issues = TestContentValidator.validate_and_fix(
            bad_test_content, enriched_context
        )

        assert not is_valid
        assert len(issues) > 0
        assert any("src." in str(issue) for issue in issues)
        assert any(issue.category == "import" for issue in issues)

    def test_orm_instantiation_validation(self):
        """Test validation of ORM model instantiation in parametrize."""
        enriched_context = {
            "packaging": {"disallowed_import_prefixes": []},
            "entities": {
                "City": {
                    "kind": "sqlalchemy.Model",
                    "instantiate_real": False,
                }
            },
            "test_safety_rules": [],
        }

        # Test bad parametrization
        bad_test_content = """
import pytest
from weather_collector.models import City

@pytest.mark.parametrize("existing_city", [None, City("X", "Y", 0, 0)])
def test_collect_weather_data(existing_city):
    pass
"""

        is_valid, issues = TestContentValidator.validate_and_fix(
            bad_test_content, enriched_context
        )

        assert not is_valid
        assert len(issues) > 0
        assert any(
            "City" in str(issue) and "parametrize" in str(issue) for issue in issues
        )
        assert any(issue.category == "instantiation" for issue in issues)

    def test_auto_fix_imports(self):
        """Test automatic fixing of import issues."""
        enriched_context = {
            "packaging": {
                "disallowed_import_prefixes": ["src."],
            },
            "imports": {
                "canonical_import": "from weather_collector.scheduler import WeatherScheduler"
            },
            "entities": {},
            "test_safety_rules": [],
        }

        bad_test_content = """
import pytest
import src.weather_collector.scheduler as scheduler_module

def test_weather_scheduler():
    ws = scheduler_module.WeatherScheduler()
    assert ws is not None
"""

        fixed_content, is_valid, remaining_issues = (
            TestContentValidator.validate_and_fix(bad_test_content, enriched_context)
        )

        # Should have attempted to fix the import
        assert "import weather_collector.scheduler" in fixed_content
        assert "import src.weather_collector.scheduler" not in fixed_content


class TestIntegrationScenario:
    """Test the complete integration scenario."""

    def test_weather_scheduler_complete_pipeline(self):
        """Test the complete pipeline that would fix the weather scheduler issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Set up the problematic project structure
            src_dir = project_root / "src"
            src_dir.mkdir()

            weather_pkg = src_dir / "weather_collector"
            weather_pkg.mkdir()
            (weather_pkg / "__init__.py").write_text("")

            scheduler_file = weather_pkg / "scheduler.py"
            scheduler_file.write_text("""
class WeatherScheduler:
    def __init__(self):
        self.api_client = None
        self.running = False
""")

            models_file = weather_pkg / "models.py"
            models_file.write_text("""
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'
    def __init__(self, name, country, latitude, longitude, timezone_offset=0):
        pass
""")

            (project_root / "pyproject.toml").write_text("""
[tool.setuptools]
package-dir = {"" = "src"}
""")

            # Step 1: Build enriched context
            builder = EnrichedContextBuilder()
            enriched_context = builder.build_enriched_context(
                source_file=scheduler_file,
                project_root=project_root,
            )

            # Step 2: Simulate problematic LLM output (like the original)
            problematic_test = """
import pytest
import src.weather_collector.scheduler as scheduler_module

@pytest.mark.parametrize("existing_city", [None, scheduler_module.City("X", "Y", 0, 0)])
def test_collect_weather_data(existing_city):
    ws = scheduler_module.WeatherScheduler()
    assert ws is not None
"""

            # Step 3: Validate and fix
            fixed_content, is_valid, issues = TestContentValidator.validate_and_fix(
                problematic_test, enriched_context
            )

            # Verify the key issues are caught
            import_issues = [issue for issue in issues if issue.category == "import"]
            instantiation_issues = [
                issue for issue in issues if issue.category == "instantiation"
            ]

            assert len(import_issues) > 0, "Should catch src. import issues"
            assert len(instantiation_issues) > 0, (
                "Should catch ORM instantiation in parametrize"
            )

            # Verify fixes are applied
            assert "import weather_collector.scheduler" in fixed_content
            assert "import src.weather_collector.scheduler" not in fixed_content

            # Step 4: Verify enriched context provides the right guidance
            formatted_context = builder.format_for_llm(enriched_context)

            # Should provide correct import statement
            assert "weather_collector.scheduler" in formatted_context

            # Should warn about disallowed prefixes
            assert "src." in formatted_context

            # Should provide safety rules
            assert (
                "Never create domain objects inside @pytest.mark.parametrize"
                in formatted_context
            )

            print("✅ Complete pipeline test passed!")
            print("📋 Issues caught:", [str(issue) for issue in issues])
            print("🔧 Fixed content preview:")
            print(fixed_content[:200] + "...")
            print("📝 Context guidance preview:")
            print(formatted_context[:300] + "...")


if __name__ == "__main__":
    # Run a quick integration test
    test = TestIntegrationScenario()
    test.test_weather_scheduler_complete_pipeline()
