"""
The agent's database tools as a Model Context Protocol (MCP) server, so the same read-only tools can
be used from Claude Code, Claude Desktop or any MCP client.

Only the read-only tools are exposed (no send_email); the connection is opened read-only as well.

    claude mcp add shopco -- python /path/to/mcp_server.py        # Claude Code
    # Claude Desktop: add {"command": "python", "args": ["/path/to/mcp_server.py"]} under "mcpServers"
"""

import os
import sys

from mcp.server.mcpserver import MCPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_db  # noqa: E402
import sql_tools  # noqa: E402

DB = os.environ.get("SHOPCO_DB", os.path.join(HERE, "data", "shopco.db"))

server = MCPServer(
    "shopco-sql",
    instructions="Read-only access to ShopCo's analytics database (customers, products, orders, order_items, "
    "support_tickets). Revenue = SUM(quantity * unit_price * (1 - discount)) over completed orders.",
)


def _call(name, args):
    if not os.path.exists(DB):
        make_db.main(["--out_dir", os.path.dirname(DB)])
    text, is_error, _ = sql_tools.run_tool(name, args, sql_tools.ToolContext(db_path=DB))
    return f"Error: {text}" if is_error else text


@server.tool()
def list_tables() -> str:
    """List every table with its row count."""
    return _call("list_tables", {})


@server.tool()
def describe_table(table: str) -> str:
    """Columns, types and three sample rows of one table."""
    return _call("describe_table", {"table": table})


@server.tool()
def run_sql(query: str) -> str:
    """Run one read-only SQLite SELECT and return up to 50 rows as tab-separated text."""
    return _call("run_sql", {"query": query})


if __name__ == "__main__":
    server.run("stdio")
