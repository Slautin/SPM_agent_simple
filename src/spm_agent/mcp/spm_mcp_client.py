from unittest import result

from spm_agent.config import SPM_MCP_SERVER_CONFIG
from langchain_mcp_adapters.client import MultiServerMCPClient

class SPMMCPClient:
    def __init__(self, server_config: dict = SPM_MCP_SERVER_CONFIG) -> None:
        self.client = MultiServerMCPClient(server_config)
        self.tools = None
        self._tool_registry = None

    async def get_tools(self,):
        if self.tools is None:
            self.tools = await self.client.get_tools()
            self._tool_registry = {tool.name: tool for tool in self.tools}
        return self.tools
    
    async def get_tool(self, tool_name: str):
        if self._tool_registry is None:
            await self.get_tools()
        
        if tool_name not in self._tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found in MCP server."
                             f" Available tools: {list(self._tool_registry.keys())}")
        else:
            return self._tool_registry.get(tool_name)
        
    async def call_tool(self, 
                        tool_name: str, 
                        tool_args: dict | None = None,):
        tool = await self.get_tool(tool_name)
        return await tool.ainvoke(tool_args or {})