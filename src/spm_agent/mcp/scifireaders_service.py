# src/spm_agent/mcp/scifireaders_service.py

import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from spm_agent.config import SCIFIREADERS_MCP_COMMAND

class SciFiReadersService:
    def __init__(
            self,
            command: str = str(SCIFIREADERS_MCP_COMMAND),
            tool_name: str = "read_scifireaders_file"
            ) -> None:
        
        self.command = command
        self.tool_name = tool_name

    async def read_file(
            self,
            file_path: str,
            return_mode: str = "data",
    ) -> dict:
        server_params = StdioServerParameters(
            command=self.command,
            args=[]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    self.tool_name,
                    {
                        "file_path": file_path,
                        "return_mode": return_mode
                    },
                )
        
        #payload_dict = json.loads(result[0]['text'])

        return result
            


