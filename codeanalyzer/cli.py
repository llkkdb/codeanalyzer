"""
Command-line interface for codeanalyzer.
"""
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .config import Config, setup_logging
from .llm.system import CodeUnderstandingSystem
from .exceptions import CodeAnalyzerError

# Initialize logging
config = Config()
logger = setup_logging(config)


@click.group()
@click.pass_context
def cli(ctx):
    """Improved Code Understanding System with Sessions"""
    try:
        ctx.obj = CodeUnderstandingSystem(config)
    except CodeAnalyzerError as e:
        click.echo(f"Error initializing system: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.argument("query")
@click.option("--k", default=5, help="Number of context documents to retrieve")
@click.pass_obj
def ask(system: CodeUnderstandingSystem, query: str, k: int):
    """Ask a question in current session with progress feedback"""
    try:
        if not system.session_manager.active_session:
            click.echo("Error: No active session!", err=True)
            click.echo("Create a new session with:", err=True)
            click.echo("  codeanalyzer session new --name my-session", err=True)
            return

        with click.progressbar(length=4, label="Processing query") as bar:
            # Generate commands
            commands = system.generate_search_commands(query)
            click.echo(f"\nGenerated commands:\n" + "\n".join(commands))
            bar.update(1)

            # Search for files
            found_files = system.execute_search(commands)
            click.echo(f"Found {len(found_files)} relevant files")
            bar.update(1)

            # Process files
            session = system.session_manager.active_session
            session.add_files(found_files)
            bar.update(1)

            # Generate answer
            answer = system.ask(query, k=k)
            bar.update(1)

        click.echo(f"\nAnswer:\n{answer}")
    except CodeAnalyzerError as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logger.error(f"Error in ask command: {e}", exc_info=True)


@cli.group()
def session():
    """Session management commands"""
    pass


@session.command(name="new")
@click.option("--name", default=None, help="Session name")
@click.option("--chunk-size", default=None, type=int,
              help="Chunk size for document processing")
@click.pass_obj
def new_session(system: CodeUnderstandingSystem, name: Optional[str], chunk_size: Optional[int]):
    """Create new session with custom parameters"""
    try:
        session_id = name or f"session_{uuid.uuid4().hex[:8]}"
        session = system.session_manager.create_session(session_id, chunk_size)
        click.echo(f"Created new session: {session.session_id}")
    except Exception as e:
        click.echo(f"Error creating session: {e}", err=True)


@session.command(name="switch")
@click.argument("session_id")
@click.pass_obj
def switch_session(system: CodeUnderstandingSystem, session_id: str):
    """Switch active session"""
    try:
        if system.session_manager.switch_session(session_id):
            click.echo(f"Switched to session: {session_id}")
        else:
            # Try to load from disk
            sess = system.session_manager.load_session(session_id)
            if sess:
                system.session_manager.active_session = sess
                click.echo(f"Loaded and switched to session: {session_id}")
            else:
                click.echo(f"Error: Session '{session_id}' not found!", err=True)
                click.echo("Run 'codeanalyzer session list' to see available sessions", err=True)
                click.echo("Or create a new session with 'codeanalyzer session new --name {}'".format(session_id), err=True)
    except Exception as e:
        click.echo(f"Error switching session: {e}", err=True)


@session.command(name="list")
@click.pass_obj
def list_sessions(system: CodeUnderstandingSystem):
    """List all sessions with additional metadata"""
    try:
        click.echo("Active sessions:")
        for session_id, sess in system.session_manager.sessions.items():
            active = (system.session_manager.active_session and
                     system.session_manager.active_session.session_id == session_id)
            files_count = len(sess.context_files)
            last_accessed = sess.last_accessed.strftime("%Y-%m-%d %H:%M")
            click.echo(
                f" {'*' if active else ' '} {session_id} - "
                f"{files_count} files - Last accessed: {last_accessed}"
            )

        # Check disk for persisted sessions
        if system.session_manager.storage_dir.exists():
            persisted = []
            for d in system.session_manager.storage_dir.iterdir():
                if (d.is_dir() and (d / "meta.json").exists() and
                        d.name not in system.session_manager.sessions):
                    try:
                        meta = json.loads((d / "meta.json").read_text())
                        created = datetime.fromisoformat(meta["created_at"]).strftime("%Y-%m-%d")
                        last_accessed = datetime.fromisoformat(
                            meta.get("last_accessed", meta["created_at"])
                        ).strftime("%Y-%m-%d")
                        files_count = len(meta.get("context_files", []))
                        persisted.append((d.name, created, last_accessed, files_count))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        persisted.append((d.name, "Unknown", "Unknown", 0))

            if persisted:
                click.echo("\nPersisted sessions (not loaded):")
                for session_id, created, last_accessed, files_count in persisted:
                    click.echo(
                        f"   {session_id} - {files_count} files - "
                        f"Created: {created} - Last accessed: {last_accessed}"
                    )
    except Exception as e:
        click.echo(f"Error listing sessions: {e}", err=True)


@session.command(name="clean")
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
@click.pass_obj
def clean_sessions(system: CodeUnderstandingSystem, force: bool):
    """Clean up old/unused sessions"""
    try:
        if not force:
            if not click.confirm("This will remove sessions older than 30 days. Continue?"):
                return

        storage_dir = system.session_manager.storage_dir
        if not storage_dir.exists():
            click.echo("No sessions to clean")
            return

        count = 0
        now = datetime.now()
        for d in storage_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                try:
                    meta = json.loads((d / "meta.json").read_text())
                    last_accessed = datetime.fromisoformat(
                        meta.get("last_accessed", meta["created_at"])
                    )

                    # If older than 30 days and not currently loaded
                    if ((now - last_accessed).days > 30 and
                            d.name not in system.session_manager.sessions):
                        # Remove the session directory
                        shutil.rmtree(d)
                        count += 1
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                    click.echo(f"Error cleaning session {d.name}: {e}")

        click.echo(f"Cleaned up {count} old sessions")
    except Exception as e:
        click.echo(f"Error cleaning sessions: {e}", err=True)


@cli.command(name="config")
@click.option("--max-file-size", type=int, help="Maximum file size in bytes")
@click.option("--add-exclude", help="Add directory or extension to exclusion list")
@click.option("--remove-exclude", help="Remove directory or extension from exclusion list")
@click.option("--list-exclude", is_flag=True, help="List excluded directories and extensions")
def configure(max_file_size: int, add_exclude: str, remove_exclude: str, list_exclude: bool) -> None:
    """Configure system parameters"""
    try:
        cfg = Config()

        # Update config
        if max_file_size:
            cfg.set("max_file_size", max_file_size)
            click.echo(f"Set maximum file size to {max_file_size} bytes")

        if add_exclude:
            if add_exclude.startswith("."):
                excluded = cfg.get("excluded_extensions", [])
                if add_exclude not in excluded:
                    excluded.append(add_exclude)
                    cfg.set("excluded_extensions", excluded)
                    click.echo(f"Added {add_exclude} to excluded extensions")
            else:
                excluded = cfg.get("excluded_dirs", [])
                if add_exclude not in excluded:
                    excluded.append(add_exclude)
                    cfg.set("excluded_dirs", excluded)
                    click.echo(f"Added {add_exclude} to excluded directories")

        if remove_exclude:
            if remove_exclude.startswith("."):
                excluded = cfg.get("excluded_extensions", [])
                if remove_exclude in excluded:
                    excluded.remove(remove_exclude)
                    cfg.set("excluded_extensions", excluded)
                    click.echo(f"Removed {remove_exclude} from excluded extensions")
            else:
                excluded = cfg.get("excluded_dirs", [])
                if remove_exclude in excluded:
                    excluded.remove(remove_exclude)
                    cfg.set("excluded_dirs", excluded)
                    click.echo(f"Removed {remove_exclude} from excluded directories")

        # List exclusions
        if list_exclude:
            click.echo("Excluded directories:")
            for d in cfg.excluded_dirs:
                click.echo(f"  - {d}")
            click.echo("Excluded extensions:")
            for e in cfg.excluded_extensions:
                click.echo(f"  - {e}")

        # Save config
        if max_file_size or add_exclude or remove_exclude:
            cfg.save()
            click.echo("Configuration saved")

    except Exception as e:
        click.echo(f"Error configuring system: {e}", err=True)


if __name__ == "__main__":
    cli()
