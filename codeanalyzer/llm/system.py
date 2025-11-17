"""
Code understanding system with LLM integration.
"""
import os
import asyncio
import time
import threading
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate

from ..core.executor import SafeCommandExecutor
from ..storage.manager import SessionManager
from ..config import Config
from ..exceptions import ConfigurationError, LLMError

logger = logging.getLogger("codeanalyzer.llm")


class CodeUnderstandingSystem:
    """
    Improved system with asynchronous processing and better caching.

    Integrates LLM-based command generation with safe execution and
    vector-based context retrieval.
    """

    def __init__(self, config: Config = None, llm=None, command_timeout: int = None):
        """
        Initialize code understanding system.

        Args:
            config: Configuration object. If None, uses defaults.
            llm: Language model instance. If None, creates ChatOpenAI instance.
            command_timeout: Command execution timeout. If None, uses config default.

        Raises:
            ConfigurationError: If OPENAI_API_KEY is not set.
        """
        self.config = config or Config()

        # IMPORTANT: Validate OPENAI_API_KEY before creating LLM
        if llm is None and not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it to your OpenAI API key. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )

        self.llm = llm if llm is not None else ChatOpenAI(
            temperature=0.1,
            model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
        self.command_executor = SafeCommandExecutor(
            self.config,
            command_timeout or self.config.command_timeout
        )
        self.session_manager = SessionManager(self.config)
        self.knowledge_guidelines = self._load_knowledge_guidelines()
        self._command_generation_cache = {}
        self._command_gen_lock = threading.Lock()

    def _load_knowledge_guidelines(self) -> str:
        """
        Load knowledge guidelines from file.

        Returns:
            Knowledge guidelines string.
        """
        try:
            guidelines_path = Path("knowledge_guidelines.md")
            if guidelines_path.exists():
                return guidelines_path.read_text()
            return "# Default Knowledge Guidelines\nGenerate precise search commands for code exploration."
        except (OSError, IOError) as e:
            logger.warning(f"Could not load knowledge guidelines: {e}")
            return "# Default Knowledge Guidelines\nGenerate precise search commands for code exploration."

    async def generate_search_commands_async(self, query: str) -> List[str]:
        """
        Generate search commands with caching.

        Args:
            query: User query for code search.

        Returns:
            List of shell commands to execute.

        Raises:
            LLMError: If command generation fails.
        """
        # Check cache first
        with self._command_gen_lock:
            cache_key = query
            if cache_key in self._command_generation_cache:
                cache_time, commands = self._command_generation_cache[cache_key]
                # Cache valid for 1 hour
                if time.time() - cache_time < 3600:
                    logger.debug("Cache hit for command generation")
                    return commands

        # Not in cache or expired, generate new commands
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.knowledge_guidelines}\n\nGenerate Linux commands to find relevant code files. Focus on precision to avoid too many results."),
            ("human", "Query: {query}\nRespond ONLY with commands separated by newlines:")
        ])

        try:
            response = await prompt.ainvoke(self.llm, {"query": query})
            commands = [cmd.strip() for cmd in response.content.split("\n") if cmd.strip()]

            # Ensure all find commands use maxdepth
            for i, cmd in enumerate(commands):
                if cmd.startswith("find") and "-maxdepth" not in cmd:
                    commands[i] = f"{cmd} -maxdepth 3"

            # Cache result
            with self._command_gen_lock:
                self._command_generation_cache[cache_key] = (time.time(), commands)

            logger.info(f"Generated {len(commands)} commands for query")
            return commands
        except Exception as e:
            error_msg = f"Error generating commands: {e}"
            logger.error(error_msg)
            raise LLMError(error_msg)

    def generate_search_commands(self, query: str) -> List[str]:
        """Synchronous wrapper for generate_search_commands_async."""
        return asyncio.run(self.generate_search_commands_async(query))

    async def execute_search_async(self, commands: List[str]) -> List[Path]:
        """
        Execute search commands in parallel.

        Args:
            commands: List of commands to execute.

        Returns:
            List of unique file paths found.
        """
        found_files = set()

        # Execute commands concurrently
        tasks = [self.command_executor.execute_async(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for cmd, result in zip(commands, results):
            if isinstance(result, Exception):
                logger.error(f"Command failed: {cmd} - {result}")
            else:
                found_files.update(result)

        logger.info(f"Found {len(found_files)} unique files")
        return list(found_files)

    def execute_search(self, commands: List[str]) -> List[Path]:
        """Synchronous wrapper for execute_search_async."""
        return asyncio.run(self.execute_search_async(commands))

    async def ask_async(self, query: str, k: int = 5) -> str:
        """
        Process a query asynchronously.

        Args:
            query: User query.
            k: Number of context documents to retrieve.

        Returns:
            Answer string from LLM.

        Raises:
            LLMError: If query processing fails.
        """
        if not self.session_manager.active_session:
            return "No active session! Create one with 'session new'"

        try:
            # Generate and execute commands in parallel
            commands = await self.generate_search_commands_async(query)
            found_files = await self.execute_search_async(commands)

            # Add files to session and get context
            session = self.session_manager.active_session
            await session.add_files_async(found_files)
            context = await session.query_context_async(query, k=k)

            # Generate answer
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{self.knowledge_guidelines}\n\nContext:\n{context}"),
                ("human", "Question: {query}")
            ])
            response = await prompt.ainvoke(self.llm, {"query": query})

            # Persist session after use
            await self.session_manager.persist_session_async(session)

            return response.content
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            raise LLMError(error_msg)

    def ask(self, query: str, k: int = 5) -> str:
        """Synchronous wrapper for ask_async."""
        return asyncio.run(self.ask_async(query, k))
