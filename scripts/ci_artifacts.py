#!/usr/bin/env python3
"""
CI Artifact Management Script for testcraft.

This script manages CI artifacts including:
1. Test reports and coverage data
2. Prompt regression artifacts
3. Documentation check results  
4. Build artifacts cleanup
5. Artifact size optimization

Run from project root: python scripts/ci_artifacts.py
"""

import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import gzip
import tarfile


class CIArtifactManager:
    """Manage CI artifacts for storage and cleanup."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the artifact manager."""
        self.project_root = project_root or Path.cwd()
        self.artifacts_base = self.project_root / "ci-artifacts"
        self.artifacts_base.mkdir(exist_ok=True)
        
    def collect_test_artifacts(self) -> Dict[str, Any]:
        """Collect and organize test artifacts."""
        print("📦 Collecting test artifacts...")
        
        artifacts = {
            "timestamp": datetime.now().isoformat(),
            "collected_files": [],
            "test_results": {},
            "coverage_data": {},
            "size_info": {}
        }
        
        # Collect pytest results
        pytest_files = list(self.project_root.glob("pytest-results*.xml"))
        for pytest_file in pytest_files:
            if pytest_file.exists():
                dest = self.artifacts_base / "test-results" / pytest_file.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(pytest_file, dest)
                artifacts["collected_files"].append(str(dest))
                artifacts["size_info"][str(dest)] = pytest_file.stat().st_size
        
        # Collect coverage reports
        coverage_dirs = ["htmlcov", "coverage_html"]
        for cov_dir in coverage_dirs:
            cov_path = self.project_root / cov_dir
            if cov_path.exists() and cov_path.is_dir():
                dest_dir = self.artifacts_base / "coverage" / cov_dir
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(cov_path, dest_dir)
                artifacts["collected_files"].append(str(dest_dir))
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
                artifacts["size_info"][str(dest_dir)] = total_size
        
        # Collect coverage.xml if it exists
        coverage_xml = self.project_root / "coverage.xml"
        if coverage_xml.exists():
            dest = self.artifacts_base / "coverage" / "coverage.xml"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(coverage_xml, dest)
            artifacts["collected_files"].append(str(dest))
            artifacts["size_info"][str(dest)] = coverage_xml.stat().st_size
        
        print(f"  ✓ Collected {len(artifacts['collected_files'])} test artifact files")
        return artifacts
    
    def collect_prompt_artifacts(self) -> Dict[str, Any]:
        """Collect prompt regression test artifacts."""
        print("📝 Collecting prompt regression artifacts...")
        
        artifacts = {
            "timestamp": datetime.now().isoformat(),
            "prompt_artifacts": [],
            "size_info": {}
        }
        
        # Look for prompt regression artifacts
        prompt_dirs = [
            self.project_root / "prompt-regression-artifacts",
            Path("prompt-regression-artifacts")  # Relative path
        ]
        
        for prompt_dir in prompt_dirs:
            if prompt_dir.exists() and prompt_dir.is_dir():
                dest_dir = self.artifacts_base / "prompt-regression"
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(prompt_dir, dest_dir)
                
                # Collect file info
                for file_path in dest_dir.rglob("*"):
                    if file_path.is_file():
                        artifacts["prompt_artifacts"].append(str(file_path))
                        artifacts["size_info"][str(file_path)] = file_path.stat().st_size
                
                print(f"  ✓ Collected {len(artifacts['prompt_artifacts'])} prompt artifact files")
                break
        else:
            print("  ✓ No prompt regression artifacts found")
        
        return artifacts
    
    def collect_security_artifacts(self) -> Dict[str, Any]:
        """Collect security scan artifacts."""
        print("🔒 Collecting security scan artifacts...")
        
        artifacts = {
            "timestamp": datetime.now().isoformat(),
            "security_files": [],
            "size_info": {}
        }
        
        # Look for security scan results
        security_files = [
            "safety-report.json",
            "bandit-report.json",
            "security-scan-results.json"
        ]
        
        for sec_file in security_files:
            sec_path = self.project_root / sec_file
            if sec_path.exists():
                dest = self.artifacts_base / "security" / sec_file
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(sec_path, dest)
                artifacts["security_files"].append(str(dest))
                artifacts["size_info"][str(dest)] = sec_path.stat().st_size
        
        print(f"  ✓ Collected {len(artifacts['security_files'])} security artifact files")
        return artifacts
    
    def compress_artifacts(self, max_size_mb: int = 50) -> Dict[str, Any]:
        """Compress large artifacts to save space."""
        print(f"🗜️  Compressing artifacts larger than {max_size_mb}MB...")
        
        compression_info = {
            "timestamp": datetime.now().isoformat(),
            "compressed_files": [],
            "size_savings": {}
        }
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for artifact_path in self.artifacts_base.rglob("*"):
            if artifact_path.is_file() and not artifact_path.name.endswith(('.gz', '.tar.gz')):
                file_size = artifact_path.stat().st_size
                
                if file_size > max_size_bytes:
                    # Compress the file
                    compressed_path = artifact_path.with_suffix(artifact_path.suffix + '.gz')
                    
                    try:
                        with open(artifact_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Remove original and track compression
                        compressed_size = compressed_path.stat().st_size
                        compression_info["compressed_files"].append(str(compressed_path))
                        compression_info["size_savings"][str(artifact_path)] = {
                            "original_size": file_size,
                            "compressed_size": compressed_size,
                            "savings_bytes": file_size - compressed_size,
                            "savings_percent": round((file_size - compressed_size) / file_size * 100, 1)
                        }
                        
                        artifact_path.unlink()  # Remove original
                        print(f"  ✓ Compressed {artifact_path.name}: {file_size // 1024}KB → {compressed_size // 1024}KB")
                        
                    except Exception as e:
                        print(f"  ⚠️  Failed to compress {artifact_path}: {e}")
        
        total_savings = sum(info["savings_bytes"] for info in compression_info["size_savings"].values())
        if total_savings > 0:
            print(f"  ✓ Total compression savings: {total_savings // 1024}KB")
        
        return compression_info
    
    def cleanup_old_artifacts(self, retention_days: int = 7) -> Dict[str, Any]:
        """Clean up old artifacts beyond retention period."""
        print(f"🧹 Cleaning up artifacts older than {retention_days} days...")
        
        cleanup_info = {
            "timestamp": datetime.now().isoformat(),
            "cleaned_files": [],
            "space_freed": 0
        }
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for artifact_path in self.artifacts_base.rglob("*"):
            if artifact_path.is_file():
                # Check file modification time
                mod_time = datetime.fromtimestamp(artifact_path.stat().st_mtime)
                
                if mod_time < cutoff_date:
                    file_size = artifact_path.stat().st_size
                    try:
                        artifact_path.unlink()
                        cleanup_info["cleaned_files"].append(str(artifact_path))
                        cleanup_info["space_freed"] += file_size
                        
                    except Exception as e:
                        print(f"  ⚠️  Failed to delete {artifact_path}: {e}")
        
        # Remove empty directories
        for dir_path in self.artifacts_base.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
        
        if cleanup_info["space_freed"] > 0:
            print(f"  ✓ Cleaned {len(cleanup_info['cleaned_files'])} files, freed {cleanup_info['space_freed'] // 1024}KB")
        else:
            print("  ✓ No old artifacts to clean up")
        
        return cleanup_info
    
    def generate_artifact_report(self) -> Dict[str, Any]:
        """Generate a comprehensive artifact report."""
        print("📊 Generating artifact report...")
        
        report = {
            "generation_time": datetime.now().isoformat(),
            "artifact_summary": {},
            "storage_info": {},
            "file_inventory": []
        }
        
        if not self.artifacts_base.exists():
            report["artifact_summary"]["status"] = "no_artifacts"
            return report
        
        # Calculate storage info
        total_size = 0
        file_count = 0
        
        for artifact_path in self.artifacts_base.rglob("*"):
            if artifact_path.is_file():
                file_size = artifact_path.stat().st_size
                total_size += file_size
                file_count += 1
                
                report["file_inventory"].append({
                    "path": str(artifact_path.relative_to(self.artifacts_base)),
                    "size": file_size,
                    "modified": datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat()
                })
        
        report["storage_info"] = {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "artifact_categories": self._categorize_artifacts()
        }
        
        report["artifact_summary"]["status"] = "artifacts_present"
        
        # Save report
        report_path = self.artifacts_base / "artifact-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ Generated artifact report: {report_path}")
        print(f"  📏 Total artifact size: {report['storage_info']['total_size_mb']}MB ({file_count} files)")
        
        return report
    
    def _categorize_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Categorize artifacts by type."""
        categories = {
            "test_results": {"pattern": ["test-results", "pytest"], "count": 0, "size": 0},
            "coverage_reports": {"pattern": ["coverage", "htmlcov"], "count": 0, "size": 0},
            "prompt_regression": {"pattern": ["prompt-regression", "prompt"], "count": 0, "size": 0},
            "security_scans": {"pattern": ["security", "safety", "bandit"], "count": 0, "size": 0},
            "other": {"pattern": [], "count": 0, "size": 0}
        }
        
        for artifact_path in self.artifacts_base.rglob("*"):
            if artifact_path.is_file():
                file_size = artifact_path.stat().st_size
                categorized = False
                
                for category, info in categories.items():
                    if category == "other":
                        continue
                    
                    if any(pattern in str(artifact_path).lower() for pattern in info["pattern"]):
                        info["count"] += 1
                        info["size"] += file_size
                        categorized = True
                        break
                
                if not categorized:
                    categories["other"]["count"] += 1
                    categories["other"]["size"] += file_size
        
        return categories
    
    def run_full_collection(self, compress: bool = True, cleanup: bool = True) -> Dict[str, Any]:
        """Run full artifact collection and management."""
        print("🚀 Starting full CI artifact collection...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "operations": []
        }
        
        # Collect all artifact types
        results["operations"].append({
            "operation": "collect_test_artifacts",
            "result": self.collect_test_artifacts()
        })
        
        results["operations"].append({
            "operation": "collect_prompt_artifacts", 
            "result": self.collect_prompt_artifacts()
        })
        
        results["operations"].append({
            "operation": "collect_security_artifacts",
            "result": self.collect_security_artifacts()
        })
        
        # Compress if requested
        if compress:
            results["operations"].append({
                "operation": "compress_artifacts",
                "result": self.compress_artifacts()
            })
        
        # Cleanup if requested
        if cleanup:
            results["operations"].append({
                "operation": "cleanup_old_artifacts",
                "result": self.cleanup_old_artifacts()
            })
        
        # Generate final report
        results["operations"].append({
            "operation": "generate_artifact_report",
            "result": self.generate_artifact_report()
        })
        
        results["end_time"] = datetime.now().isoformat()
        
        # Save full results
        results_path = self.artifacts_base / "collection-results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Artifact collection complete! Results saved to: {results_path}")
        return results


def main():
    """Main entry point for the artifact manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage testcraft CI artifacts")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect artifacts, don't compress or cleanup"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip compression of large files"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true", 
        help="Skip cleanup of old artifacts"
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=7,
        help="Retention period for artifacts in days (default: 7)"
    )
    
    args = parser.parse_args()
    
    manager = CIArtifactManager(project_root=args.project_root)
    
    if args.collect_only:
        # Just collect artifacts
        manager.collect_test_artifacts()
        manager.collect_prompt_artifacts() 
        manager.collect_security_artifacts()
        manager.generate_artifact_report()
    else:
        # Full collection with optional compression and cleanup
        manager.run_full_collection(
            compress=not args.no_compress,
            cleanup=not args.no_cleanup
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
