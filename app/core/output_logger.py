"""
Output Logger Module

Provides stdout and stderr capture functionality for evaluation workflows.
Implements size-limited buffering with truncation support for safe memory usage.
"""

import sys
import io
from typing import Optional, TextIO
from contextvars import ContextVar


class TruncatingStringIO:
    """
    A StringIO wrapper that enforces size limits and handles truncation.
    
    When the buffer exceeds the maximum size, it truncates and adds a notice.
    """
    
    def __init__(self, max_size: int):
        """
        Initialize TruncatingStringIO with a maximum buffer size.
        
        Args:
            max_size: Maximum buffer size in bytes
        """
        self._buffer = io.StringIO()
        self._max_size = max_size
        self._current_size = 0
        self._truncated = False
    
    def write(self, s: str) -> int:
        """
        Write string to buffer, truncating if size limit is exceeded.
        
        Args:
            s: String to write
            
        Returns:
            Number of characters written
        """
        if not s:
            return 0
        
        s_bytes = len(s.encode('utf-8'))
        
        # If already truncated, ignore further writes
        if self._truncated:
            return len(s)
        
        # Check if this write would exceed the limit
        if self._current_size + s_bytes > self._max_size:
            # Calculate how much we can still write
            remaining = self._max_size - self._current_size
            
            if remaining > 0:
                # Write partial content that fits
                truncated_s = s
                while len(truncated_s.encode('utf-8')) > remaining and truncated_s:
                    truncated_s = truncated_s[:-1]
                
                if truncated_s:
                    self._buffer.write(truncated_s)
                    self._current_size += len(truncated_s.encode('utf-8'))
            
            # Add truncation notice
            truncation_notice = f"\n[OUTPUT TRUNCATED: exceeded {self._max_size // 1024}KB limit]"
            self._buffer.write(truncation_notice)
            self._truncated = True
            self._current_size += len(truncation_notice.encode('utf-8'))
            
            return len(s)
        
        # Normal write within limits
        self._buffer.write(s)
        self._current_size += s_bytes
        return len(s)
    
    def getvalue(self) -> str:
        """
        Get the current buffer contents.
        
        Returns:
            Buffer contents as string
        """
        return self._buffer.getvalue()
    
    def is_truncated(self) -> bool:
        """
        Check if truncation has occurred.
        
        Returns:
            True if buffer was truncated, False otherwise
        """
        return self._truncated
    
    def flush(self):
        """Flush the buffer (no-op for StringIO)."""
        pass


class ContextAwareStream:
    """
    A stream wrapper that routes writes to the appropriate context-specific buffer.
    
    This enables concurrent async tasks to each have their own output capture
    without cross-contamination. When a write occurs, it checks the current
    context and routes to the appropriate logger's buffer.
    """
    
    def __init__(self, original_stream: TextIO, context_var: ContextVar, stream_type: str):
        """
        Initialize context-aware stream.
        
        Args:
            original_stream: The original sys.stdout or sys.stderr
            context_var: ContextVar that holds the active OutputLogger
            stream_type: Either 'stdout' or 'stderr'
        """
        self._original_stream = original_stream
        self._context_var = context_var
        self._stream_type = stream_type
    
    def write(self, s: str) -> int:
        """
        Write to the appropriate buffer based on current context.
        
        Args:
            s: String to write
            
        Returns:
            Number of characters written
        """
        # Get the active logger from context
        active_logger = self._context_var.get(None)
        
        if active_logger is not None and active_logger._capturing:
            # Route to the logger's buffer
            if self._stream_type == 'stdout' and active_logger._stdout_buffer:
                # Write to buffer
                active_logger._stdout_buffer.write(s)
                # Also write to original stream for pytest capture
                try:
                    self._original_stream.write(s)
                    self._original_stream.flush()
                except:
                    pass
                return len(s)
            elif self._stream_type == 'stderr' and active_logger._stderr_buffer:
                # Write to buffer
                active_logger._stderr_buffer.write(s)
                # Also write to original stream for pytest capture
                try:
                    self._original_stream.write(s)
                    self._original_stream.flush()
                except:
                    pass
                return len(s)
        
        # No active logger, write to original stream
        return self._original_stream.write(s)
    
    def flush(self):
        """Flush the stream."""
        # Get the active logger from context
        active_logger = self._context_var.get(None)
        
        if active_logger is not None:
            # Flush the logger's buffer
            if self._stream_type == 'stdout' and active_logger._stdout_buffer:
                active_logger._stdout_buffer.flush()
            elif self._stream_type == 'stderr' and active_logger._stderr_buffer:
                active_logger._stderr_buffer.flush()
        else:
            # No active logger, flush original stream
            self._original_stream.flush()
    
    def __getattr__(self, name):
        """Delegate other attributes to the original stream."""
        return getattr(self._original_stream, name)


class OutputLogger:
    """
    Context manager for capturing stdout and stderr during evaluation workflows.
    
    Provides async-safe output capture with configurable size limits and
    automatic stream restoration. Uses contextvars for request isolation in
    concurrent async environments.
    
    Each OutputLogger instance maintains its own buffers and uses context
    variables to ensure output from concurrent requests doesn't get mixed.
    """
    
    # Context variables for async-safe operation - stores the active logger per context
    _active_logger: ContextVar[Optional['OutputLogger']] = ContextVar('active_output_logger', default=None)
    
    # Class-level flag to track if we've installed the context-aware streams
    _streams_installed = False
    _original_stdout: Optional[TextIO] = None
    _original_stderr: Optional[TextIO] = None
    
    @classmethod
    def _install_context_aware_streams(cls):
        """Install context-aware stream wrappers (once per process)."""
        if not cls._streams_installed:
            cls._original_stdout = sys.stdout
            cls._original_stderr = sys.stderr
            
            sys.stdout = ContextAwareStream(cls._original_stdout, cls._active_logger, 'stdout')  # type: ignore
            sys.stderr = ContextAwareStream(cls._original_stderr, cls._active_logger, 'stderr')  # type: ignore
            
            cls._streams_installed = True
    
    def __init__(
        self,
        max_stdout_size: int = 100 * 1024,  # 100KB default
        max_stderr_size: int = 50 * 1024    # 50KB default
    ):
        """
        Initialize OutputLogger with buffer size limits.
        
        Args:
            max_stdout_size: Maximum stdout buffer size in bytes (default: 100KB)
            max_stderr_size: Maximum stderr buffer size in bytes (default: 50KB)
        """
        self._max_stdout_size = max_stdout_size
        self._max_stderr_size = max_stderr_size
        
        # Buffers for this instance
        self._stdout_buffer: Optional[TruncatingStringIO] = None
        self._stderr_buffer: Optional[TruncatingStringIO] = None
        
        # Track if we're currently capturing
        self._capturing = False
        
        # Token for context variable cleanup
        self._context_token = None
    
    def __enter__(self) -> 'OutputLogger':
        """
        Start capturing stdout and stderr.
        
        Sets this logger as the active logger in the current context and
        redirects stdout/stderr to internal buffers.
        
        Returns:
            Self for context manager protocol
        """
        if self._capturing:
            raise RuntimeError("OutputLogger is already capturing")
        
        # Ensure context-aware streams are installed
        self._install_context_aware_streams()
        
        # Create new buffers
        self._stdout_buffer = TruncatingStringIO(self._max_stdout_size)
        self._stderr_buffer = TruncatingStringIO(self._max_stderr_size)
        
        # Set this logger as active in the current context
        self._context_token = self._active_logger.set(self)
        
        self._capturing = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop capturing and restore original streams.
        
        If an exception occurred, capture it in stderr. Ensures proper cleanup
        of context variables.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if not self._capturing:
            return
        
        try:
            # If an exception occurred, capture it in stderr
            if exc_type is not None and exc_val is not None:
                import traceback
                error_msg = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
                if self._stderr_buffer:
                    self._stderr_buffer.write(f"\nException occurred:\n{error_msg}")
        finally:
            # Clear context variable using the token
            if self._context_token is not None:
                self._active_logger.reset(self._context_token)
                self._context_token = None
            
            self._capturing = False
    
    def get_stdout(self) -> str:
        """
        Get captured stdout content.
        
        Returns:
            Captured stdout as string, empty string if no capture occurred
        """
        if self._stdout_buffer is None:
            return ""
        return self._stdout_buffer.getvalue()
    
    def get_stderr(self) -> str:
        """
        Get captured stderr content.
        
        Returns:
            Captured stderr as string, empty string if no capture occurred
        """
        if self._stderr_buffer is None:
            return ""
        return self._stderr_buffer.getvalue()
    
    def reset(self) -> None:
        """
        Clear all captured content from buffers.
        
        Creates new empty buffers, discarding previous content.
        """
        if self._capturing:
            # If currently capturing, create new buffers
            # The context-aware streams will automatically route to these new buffers
            self._stdout_buffer = TruncatingStringIO(self._max_stdout_size)
            self._stderr_buffer = TruncatingStringIO(self._max_stderr_size)
        else:
            # If not capturing, just clear the buffers
            self._stdout_buffer = None
            self._stderr_buffer = None
    
    def is_stdout_truncated(self) -> bool:
        """
        Check if stdout was truncated.
        
        Returns:
            True if stdout buffer was truncated, False otherwise
        """
        if self._stdout_buffer is None:
            return False
        return self._stdout_buffer.is_truncated()
    
    def is_stderr_truncated(self) -> bool:
        """
        Check if stderr was truncated.
        
        Returns:
            True if stderr buffer was truncated, False otherwise
        """
        if self._stderr_buffer is None:
            return False
        return self._stderr_buffer.is_truncated()
