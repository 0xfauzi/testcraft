"""Main CLI entry point for testcraft."""

import click


@click.group()
def app() -> None:
    """TestCraft - AI-powered test generation tool for Python projects."""
    pass


@click.command()
def version() -> None:
    """Show version information."""
    click.echo("TestCraft version 0.1.0")


app.add_command(version)


if __name__ == "__main__":
    app()
