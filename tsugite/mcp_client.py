"""MCP client using official Python SDK.

Connects to MCP servers (stdio or HTTP) and loads their tools.
"""

from typing import List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from tsugite.core.tools import Tool
from tsugite.mcp_config import MCPServerConfig


class MCPClient:
    """Client for connecting to MCP servers.

    Handles both stdio and HTTP transports.

    Example:
        # Create config
        config = MCPServerConfig(
            name="basic-memory",
            command="uvx",
            args=["basic-memory-mcp"]
        )

        # Connect and load tools
        client = MCPClient(config)
        await client.connect()
        tools = await client.get_tools()

        # Use tools in agent
        agent = TsugiteAgent(model="...", tools=tools)
    """

    def __init__(self, server_config: MCPServerConfig):
        """Initialize MCP client.

        Args:
            server_config: Configuration for MCP server
        """
        self.config = server_config
        self.session = None
        self.session_ctx = None
        self.mcp_tools = []  # Raw MCP tool objects
        self.transport = None

    async def connect(self):
        """Connect to MCP server and initialize session.

        Must be called before get_tools().
        """
        if self.config.is_stdio():
            # Connect via stdio (command-line server)
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env,
            )

            # Start stdio transport
            transport_ctx = stdio_client(server_params)
            self.transport = transport_ctx
            read, write = await transport_ctx.__aenter__()

        elif self.config.is_http():
            # Connect via HTTP
            transport_ctx = streamablehttp_client(self.config.url)
            self.transport = transport_ctx
            read, write, _ = await transport_ctx.__aenter__()
        else:
            raise ValueError(f"Unknown transport type: {self.config.type}")

        # Create session
        self.session_ctx = ClientSession(read, write)
        self.session = await self.session_ctx.__aenter__()

        # Initialize connection
        await self.session.initialize()

        # Load available tools
        response = await self.session.list_tools()
        self.mcp_tools = response.tools

    async def disconnect(self):
        """Disconnect from MCP server and cleanup resources.

        Closes the session and transport, preventing resource leaks.
        Should be called when done using the client.
        """
        # Exit session context manager
        if self.session_ctx:
            try:
                await self.session_ctx.__aexit__(None, None, None)
            except Exception:
                pass  # Best effort cleanup
            self.session_ctx = None
            self.session = None

        # Exit transport context manager
        if self.transport:
            try:
                await self.transport.__aexit__(None, None, None)
            except Exception:
                pass  # Best effort cleanup
            self.transport = None

    async def get_tools(self, allowed_tools: Optional[List[str]] = None) -> List[Tool]:
        """Get tools from MCP server as Tool objects.

        Args:
            allowed_tools: If provided, only return these tools

        Returns:
            List[Tool]: Tools wrapped in our Tool interface
        """
        tools = []

        for mcp_tool in self.mcp_tools:
            # Filter if needed
            if allowed_tools and mcp_tool.name not in allowed_tools:
                continue

            # Convert to our Tool format
            tool = Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                parameters=mcp_tool.inputSchema or {},  # Already JSON schema!
                function=self._create_tool_caller(mcp_tool.name),
            )
            tools.append(tool)

        return tools

    def _create_tool_caller(self, tool_name: str):
        """Create async function that calls MCP tool.

        Returns a function that agents can call.
        """

        async def call_mcp_tool(**kwargs):
            """Call MCP tool and return result."""
            result = await self.session.call_tool(tool_name, arguments=kwargs)

            # Parse result content
            # MCP returns list of content items
            if result.content:
                # Handle different content types
                content_items = []
                for item in result.content:
                    if item.type == "text":
                        content_items.append(item.text)
                    elif item.type == "image":
                        content_items.append(f"[Image: {item.data}]")
                    # Add other types as needed

                # Return combined content
                return "\n".join(content_items) if content_items else None

            return None

        return call_mcp_tool


async def load_mcp_tools(
    server_config: MCPServerConfig, allowed_tools: Optional[List[str]] = None
) -> tuple[MCPClient, List[Tool]]:
    """Connect to MCP server and load tools.

    Returns both the client and tools. The client must remain alive for tools
    to work, so caller should call client.disconnect() when done with the tools.

    Args:
        server_config: MCP server configuration
        allowed_tools: Optional list of tool names to load

    Returns:
        tuple[MCPClient, List[Tool]]: Client and tools from MCP server

    Example:
        from tsugite.mcp_config import load_mcp_config

        # Load config
        config = load_mcp_config()
        server_config = config["basic-memory"]

        # Load tools and keep client alive
        client, tools = await load_mcp_tools(server_config)

        # Use tools...

        # Cleanup when done
        await client.disconnect()
    """
    client = MCPClient(server_config)
    await client.connect()
    tools = await client.get_tools(allowed_tools)
    return client, tools
