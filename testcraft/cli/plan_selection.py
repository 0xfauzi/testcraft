"""Interactive plan selection utilities."""

from pathlib import Path

import click


def interactive_plan_selection(session, ui_adapter) -> list[str]:
    """Interactive plan selection with comprehensive options."""
    selected_keys = []
    
    # First, ask about overall approach
    ui_adapter.display_info(f"Found {len(session.items)} test plans ready for selection", "Plan Selection")
    
    if len(session.items) == 0:
        return selected_keys
    
    # Show selection options
    selection_options = [
        "Accept all plans",
        "Select by file", 
        "Select by element type",
        "Individual selection",
        "Accept none"
    ]
    
    ui_adapter.console.print("\n[bold cyan]Selection Options:[/]")
    for i, option in enumerate(selection_options, 1):
        ui_adapter.console.print(f"  {i}. {option}")
    
    while True:
        try:
            choice = click.prompt(
                "\nChoose selection method",
                type=click.IntRange(1, len(selection_options)),
                default=1
            )
            break
        except click.Abort:
            ui_adapter.display_info("Selection cancelled", "Cancelled")
            return []
    
    if choice == 1:  # Accept all
        selected_keys = []
        for item in session.items:
            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
            source_file = ""
            for tag in item.tags:
                if tag.startswith("source_file:"):
                    source_file = tag.replace("source_file:", "")
                    break
            key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
            selected_keys.append(key)
        ui_adapter.display_success(f"Selected all {len(selected_keys)} plans", "All Selected")
        
    elif choice == 2:  # Select by file
        selected_keys = _select_by_file(session, ui_adapter)
        
    elif choice == 3:  # Select by element type
        selected_keys = _select_by_type(session, ui_adapter)
        
    elif choice == 4:  # Individual selection
        selected_keys = _select_individual_plans(session, ui_adapter)
        
    elif choice == 5:  # Accept none
        ui_adapter.display_info("No plans selected", "None Selected")
        selected_keys = []
    
    return selected_keys


def _select_by_file(session, ui_adapter) -> list[str]:
    """Select plans by file."""
    # Group plans by file
    files_with_plans = {}
    for item in session.items:
        file_path = "unknown"
        for tag in item.tags:
            if tag.startswith("source_file:"):
                file_path = tag.replace("source_file:", "")
                break
        
        if file_path not in files_with_plans:
            files_with_plans[file_path] = []
        files_with_plans[file_path].append(item)
    
    selected_keys = []
    
    ui_adapter.console.print(f"\n[bold cyan]Files with plans ({len(files_with_plans)}):[/]")
    for i, (file_path, items) in enumerate(files_with_plans.items(), 1):
        file_name = Path(file_path).name if file_path != "unknown" else "unknown"
        ui_adapter.console.print(f"  {i}. {file_name} ({len(items)} elements)")
    
    ui_adapter.console.print("  a. Select all files")
    ui_adapter.console.print("  n. Select none")
    
    while True:
        selection = click.prompt(
            "\nEnter file numbers (comma-separated), 'a' for all, or 'n' for none",
            type=str,
            default="a"
        ).strip().lower()
        
        if selection == 'a':
            # Select all files
            for items in files_with_plans.values():
                for item in items:
                    element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
                    source_file = ""
                    for tag in item.tags:
                        if tag.startswith("source_file:"):
                            source_file = tag.replace("source_file:", "")
                            break
                    key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                    selected_keys.append(key)
            break
        elif selection == 'n':
            break
        else:
            try:
                # Parse comma-separated numbers
                file_indices = [int(x.strip()) for x in selection.split(",")]
                file_list = list(files_with_plans.items())
                
                for idx in file_indices:
                    if 1 <= idx <= len(file_list):
                        _, items = file_list[idx - 1]
                        for item in items:
                            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
                            source_file = ""
                            for tag in item.tags:
                                if tag.startswith("source_file:"):
                                    source_file = tag.replace("source_file:", "")
                                    break
                            key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                            selected_keys.append(key)
                break
            except (ValueError, IndexError):
                ui_adapter.console.print("[red]Invalid selection. Please try again.[/]")
    
    ui_adapter.display_success(f"Selected {len(selected_keys)} plans from files", "File Selection")
    return selected_keys


def _select_by_type(session, ui_adapter) -> list[str]:
    """Select plans by element type."""
    # Group plans by type
    types_with_plans = {}
    for item in session.items:
        element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
        if element_type not in types_with_plans:
            types_with_plans[element_type] = []
        types_with_plans[element_type].append(item)
    
    selected_keys = []
    
    ui_adapter.console.print(f"\n[bold cyan]Element types with plans ({len(types_with_plans)}):[/]")
    for i, (element_type, items) in enumerate(types_with_plans.items(), 1):
        ui_adapter.console.print(f"  {i}. {element_type} ({len(items)} elements)")
    
    ui_adapter.console.print("  a. Select all types")
    ui_adapter.console.print("  n. Select none")
    
    while True:
        selection = click.prompt(
            "\nEnter type numbers (comma-separated), 'a' for all, or 'n' for none",
            type=str,
            default="a"
        ).strip().lower()
        
        if selection == 'a':
            # Select all types
            for items in types_with_plans.values():
                for item in items:
                    element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
                    source_file = ""
                    for tag in item.tags:
                        if tag.startswith("source_file:"):
                            source_file = tag.replace("source_file:", "")
                            break
                    key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                    selected_keys.append(key)
            break
        elif selection == 'n':
            break
        else:
            try:
                # Parse comma-separated numbers
                type_indices = [int(x.strip()) for x in selection.split(",")]
                type_list = list(types_with_plans.items())
                
                for idx in type_indices:
                    if 1 <= idx <= len(type_list):
                        _, items = type_list[idx - 1]
                        for item in items:
                            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
                            source_file = ""
                            for tag in item.tags:
                                if tag.startswith("source_file:"):
                                    source_file = tag.replace("source_file:", "")
                                    break
                            key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                            selected_keys.append(key)
                break
            except (ValueError, IndexError):
                ui_adapter.console.print("[red]Invalid selection. Please try again.[/]")
    
    ui_adapter.display_success(f"Selected {len(selected_keys)} plans by type", "Type Selection")
    return selected_keys


def _select_individual_plans(session, ui_adapter) -> list[str]:
    """Individual plan selection with checkboxes."""
    selected_keys = []
    
    ui_adapter.console.print(f"\n[bold cyan]Individual Plan Selection ({len(session.items)} plans):[/]")
    ui_adapter.console.print("[muted]Enter plan numbers to toggle selection, 'done' when finished, 'all' to select all, 'none' to clear[/]")
    
    # Create a selection state
    selected_indices = set()
    
    while True:
        # Display current selection state
        ui_adapter.console.print(f"\n[bold]Current selection: {len(selected_indices)}/{len(session.items)} plans[/]")
        
        for i, item in enumerate(session.items, 1):
            # Extract file path for display
            file_path = "unknown"
            for tag in item.tags:
                if tag.startswith("source_file:"):
                    file_path = tag.replace("source_file:", "")
                    break
            
            # Show selection state
            checkbox = "☑️" if i in selected_indices else "☐"
            confidence_str = f"({item.confidence:.2f})" if item.confidence else ""
            file_name = Path(file_path).name if file_path != "unknown" else "unknown"
            
            summary = item.plan_summary[:50] + "..." if len(item.plan_summary) > 50 else item.plan_summary
            
            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
            ui_adapter.console.print(
                f"  {checkbox} {i:2d}. [cyan]{file_name}[/]:[green]{item.element.name}[/] "
                f"[yellow]({element_type})[/] {confidence_str}\n"
                f"       [muted]{summary}[/]"
            )
        
        # Get user input
        user_input = click.prompt(
            "\nEnter numbers to toggle, 'done', 'all', 'none', or 'help'",
            type=str,
            default="done"
        ).strip().lower()
        
        if user_input == "done":
            break
        elif user_input == "all":
            selected_indices = set(range(1, len(session.items) + 1))
        elif user_input == "none":
            selected_indices.clear()
        elif user_input == "help":
            ui_adapter.console.print(
                "\n[bold]Help:[/]\n"
                "  • Enter numbers (1,2,3) to toggle plan selection\n"
                "  • 'all' - select all plans\n"
                "  • 'none' - clear all selections\n"
                "  • 'done' - finish selection\n"
                "  • Plans with ☑️ are selected, ☐ are not selected"
            )
        else:
            try:
                # Parse numbers and toggle selection
                numbers = [int(x.strip()) for x in user_input.split(",")]
                for num in numbers:
                    if 1 <= num <= len(session.items):
                        if num in selected_indices:
                            selected_indices.remove(num)
                        else:
                            selected_indices.add(num)
                    else:
                        ui_adapter.console.print(f"[red]Invalid plan number: {num}[/]")
            except ValueError:
                ui_adapter.console.print("[red]Invalid input. Enter numbers separated by commas, or 'help' for options.[/]")
    
    # Convert selected indices to keys
    for idx in selected_indices:
        if 1 <= idx <= len(session.items):
            item = session.items[idx - 1]
            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
            source_file = ""
            for tag in item.tags:
                if tag.startswith("source_file:"):
                    source_file = tag.replace("source_file:", "")
                    break
            key = f"{source_file}::{element_type}::{item.element.name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
            selected_keys.append(key)
    
    ui_adapter.display_success(f"Selected {len(selected_keys)} plans individually", "Individual Selection")
    return selected_keys


def process_accept_patterns(session, patterns: str) -> list[str]:
    """Process accept patterns to select specific plans."""
    # Simple pattern matching - could be enhanced
    selected = []
    pattern_list = [p.strip() for p in patterns.split(",")]
    
    for item in session.items:
        element_name = item.element.name
        element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
        
        for pattern in pattern_list:
            if pattern in element_name or pattern in element_type:
                # Build canonical plannable key including source file path
                source_file = ""
                for tag in item.tags:
                    if tag.startswith("source_file:"):
                        source_file = tag.replace("source_file:", "")
                        break
                key = f"{source_file}::{element_type}::{element_name}::{item.element.line_range[0]}-{item.element.line_range[1]}"
                selected.append(key)
                break
    
    return selected
