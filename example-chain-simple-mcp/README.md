# Example Chain with a Simple MCP Server

This example shows how to embed an **MCP (Model Context Protocol) server** in a GenAI chain. The chain exposes two utility tools that AI assistants and other MCP-compatible clients can call directly, without going through the chain's main invoke endpoint.

This example is the practical counterpart of the MCP server section in the [GenAI Core developer guide](https://github.com/Stratio/genai-core/pull/688).

## What this example demonstrates

- Overriding `mcp_server()` from `BaseGenAiChain` to embed an MCP server.
- Registering MCP tools with `@mcp.tool()` and `@mcp_tool_wrapper()`.
- Using `MCP_READ_ROLES` for role-based access control on MCP tools.
- Defining typed MCP tool responses extending `McpToolResponse`.
- Structuring tool registration in a dedicated `mcp_tools.py` module.

### MCP tools exposed

| Tool | Calls chain? | Description |
|---|---|---|
| `get_current_time` | No | Returns the current UTC date and time in ISO 8601 format. |
| `format_text` | No | Formats a string: `upper`, `lower`, `title`, or `reverse`. |
| `analyze_text` | **Yes** | Returns word count and character count by delegating to `chain.chain().invoke()`. |
| `mad_hatter` | Uses an actor | Answers a request with a rhyming riddle, as the Mad Hatter. Only registered when `gateway_endpoint` is configured. |
| `run_opensearch_query` | Uses OpenSearch | Runs an OpenSearch Query DSL search on an index and returns the hits. Only registered when `opensearch_url` is configured. |

`get_current_time` and `format_text` are pure utility functions independent of the chain logic.
`analyze_text` shows the pattern of calling the chain from an MCP tool.
`mad_hatter` shows the pattern of calling a Stratio actor (an LLM) from an MCP tool.

### Using a Stratio actor from an MCP tool (`mad_hatter`)

The `mad_hatter` tool reuses the `BasicExampleActor` from [example-chain-basic-actor](../example-chain-basic-actor),
copied here as `MadHatterActor` in `simple_mcp_chain/actors/mad_hatter_actor.py`. It reproduces the three steps of
that chain (`_extract_genai_auth | _extract_user_name | _invoke_actor`) inside the tool:

1. MCP tools receive a `GenAiMetadata` (from the MCP request `_meta`) instead of a `RunnableConfig`.
   `mcp_metadata_to_runnable_config(metadata)` converts it.
2. `GenAiAuthRunnable().invoke({}, config)` builds the `GenAiAuth` of the nominal user.
3. The auth is stored in `config["metadata"]["genai_auth"]`, so the actor calls Stratio GenAI LiteLLM
   on behalf of the user instead of with the chain identity.
4. The actor input (`user_request` + `username`) is built and `actor.get_chain().invoke(actor_input, config)` runs it.

The response extends `McpCheckpointToolResponse` because LLM calls can be slow: when the server has the
checkpoint-timeout fallback enabled, the client gets a `task_id` to poll with `get_mcp_task_result`.

To enable it, set `gateway_endpoint` (the LiteLLM model group ID) in the chain config, as in `main.py`.
This requires access to Stratio GenAI LiteLLM (e.g. through the GenAI development proxy). Without it the
tool is not registered and the rest of the example works without any LLM.

### Running OpenSearch queries (`run_opensearch_query`)

Set `opensearch_url` in the chain config to enable it. The chain connects to OpenSearch as
[example-chain-opensearch](../example-chain-opensearch) does (certificates from Vault, or from the
`VAULT_LOCAL_*` environment variables locally) and fails to start if it cannot connect.

- Locally, `main.py` reads the `OPENSEARCH_URL` environment variable, which points to the developer proxy
  (e.g. `https://genai-developer-proxy-loadbalancer.<tenant>-genai.<domain>:8080/service/opensearch/`).
- When deployed in GenAI API, use the internal URL (e.g. `https://opensearch-genai.stratio-genai:9200`).

## Local deployment

To set up the chain locally, follow the steps in the [main README of this repository](../README.md). Here is a summary:

1. Make sure you have Python >= 3.11 and Poetry >= 2.2 installed.

2. Edit `pyproject.toml` and set the `stratio-releases` repository URL to your *Stratio GenAI Developer Proxy* Load Balancer:

```toml
[[tool.poetry.source]]
name = "stratio-releases"
url = "https://genai-developer-proxy-loadbalancer.your-tenant-genai.yourdomain.com:8080/service/genai-api/v1/pypi/simple/"
priority = "supplemental"
```

3. Install dependencies:

```bash
poetry config virtualenvs.in-project true
poetry config certificates.stratio-releases.cert /path/to/your/cert/folder/ca-cert.crt
poetry lock
poetry install
```

4. Run the chain (no LiteLLM connection needed unless `gateway_endpoint` is set, which enables `mad_hatter`):

```bash
poetry run python simple_mcp_chain/main.py
```

Open the Swagger UI at `http://127.0.0.1:8080/`.

### Invoke endpoint

Call `POST /invoke` with:

```json
{
  "input": {
    "text": "Hello, MCP world!"
  }
}
```

Expected response:

```json
{
  "output": {
    "text": "Hello, MCP world!",
    "word_count": 3,
    "char_count": 18
  }
}
```

### MCP tools

The MCP server is mounted at `http://127.0.0.1:8080/mcp` using the [Streamable HTTP transport](https://modelcontextprotocol.io/docs/concepts/transports).

#### Python client (recommended)

Use the `fastmcp` library (already a transitive dependency of `genai-core`):

```python
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

MCP_URL = "http://127.0.0.1:8080/mcp"

# Auth metadata required by the Stratio GenAI platform.
# Locally the server accepts any role; in production this is injected automatically.
AUTH_META = {
    "genai_state": {
        "client_auth_type": "mtls",
        "client_user_id": "<your-user>",
        "client_tenant": "<your-tenant>",
        "client_role": "consumer",
    }
}

async def main():
    async with Client(StreamableHttpTransport(MCP_URL)) as client:

        # Discover available tools
        tools = await client.list_tools()
        print([t.name for t in tools])
        # ['get_current_time', 'format_text']

        # Call get_current_time (no arguments)
        result = await client.call_tool("get_current_time", {}, meta=AUTH_META)
        print(result.structured_content)
        # {'time': '2025-05-26T10:30:00.123456+00:00', 'timezone': 'UTC'}

        # Call format_text
        result = await client.call_tool(
            "format_text",
            {"text": "hello world", "style": "title"},
            meta=AUTH_META,
        )
        print(result.structured_content)
        # {'result': 'Hello World', 'style': 'title'}

asyncio.run(main())
```

A ready-to-run version of this script is available at [`scripts/mcp_client_example.py`](scripts/mcp_client_example.py).

#### Raw JSON-RPC (curl)

The MCP Streamable HTTP transport uses JSON-RPC 2.0 over HTTP POST. It requires
**session management**: the server returns an `mcp-session-id` header on `initialize`,
and every subsequent request must include that header. Responses arrive as
Server-Sent Events (`event: message\ndata: {...}`).

The Python client handles all of this automatically. Use curl only for debugging.

```bash
# Step 1 — initialize and capture the session ID
SESSION_ID=$(curl -s -i -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  | grep -i "mcp-session-id" | tr -d '\r' | awk -F': ' '{print $2}')

echo "Session: $SESSION_ID"

# Step 2 — send the required initialized notification (no response expected)
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'

# Step 3 — list available tools
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Step 4 — call get_current_time (no user arguments)
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_current_time","arguments":{},"_meta":{"genai_state":{"client_user_id":"<your-user>","client_tenant":"<your-tenant>","client_role":"consumer"}}}}'

# Step 5 — call format_text with arguments
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"format_text","arguments":{"text":"hello world","style":"title"},"_meta":{"genai_state":{"client_user_id":"<your-user>","client_tenant":"<your-tenant>","client_role":"consumer"}}}}'
```

Expected SSE response for `format_text`:

```
event: message
data: {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"{\"result\":\"Hello World\",\"style\":\"title\"}"}],"structuredContent":{"result":"Hello World","style":"title"},"isError":false}}
```

##### Calling `mad_hatter` with curl

`mad_hatter` is only listed by `tools/list` when the chain runs with `gateway_endpoint` set (see `main.py`).
Unlike the utility tools, it calls Stratio GenAI LiteLLM on behalf of the user in `_meta.genai_state`,
so it needs a **real** user and tenant, not placeholders. The `local-env` script (see the
[main README](../README.md)) writes them to `.local_env/genai-env.sh`. Load that file in the shell where you run curl,
then reuse the `SESSION_ID` from Step 1:

```bash
# Run from the example-chain-simple-mcp folder (or use the absolute path to the file)
source .local_env/genai-env.sh

# Step 6 — call mad_hatter (the user and tenant come from the env vars loaded above)
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"mad_hatter","arguments":{"user_request":"Hi! Nice to meet you! Where is the Queen of Hearts?"},"_meta":{"genai_state":{"client_auth_type":"mtls","client_user_id":"'"$GENAI_METADATA_USER_ID"'","client_tenant":"'"$GENAI_METADATA_TENANT"'","client_role":"'"$GENAI_METADATA_ROLE"'"}}}}'
```

The `structuredContent` of the response contains `user_request_explanation`, `riddle` and `riddle_explanation`.

LLM calls can be slow. If the tool exceeds the server's checkpoint timeout, the response contains a `task_id`
instead of the riddle. Poll `get_mcp_task_result` until `status` is `done`, `error` or `not_found`:

```bash
# Step 7 — (only if Step 6 returned a task_id) poll for the result
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"get_mcp_task_result","arguments":{"task_id":"<task_id>"},"_meta":{"genai_state":{"client_user_id":"'"$GENAI_METADATA_USER_ID"'","client_tenant":"'"$GENAI_METADATA_TENANT"'","client_role":"'"$GENAI_METADATA_ROLE"'"}}}}'
```

If the response is `{"error": "Mad Hatter actor failed: ..."}`, the call to LiteLLM failed: check the
LiteLLM variables in `.local_env/genai-env.env`.

> **Note**: the `_meta.genai_state.client_role` field is required by `mcp_tool_wrapper`
> for role-based access control. Without it the tool returns
> `{"error": "Access denied: user role not provided in request metadata."}`.
> The Python client passes this via the `meta=` argument to `call_tool()`.

## Deployment in Stratio GenAI API

1. Build the chain package:

```bash
poetry build
```

2. Open the Swagger UI of the Stratio GenAI API installed in your development environment.

3. Upload the package with `POST /v1/packages`.

4. Deploy the chain with `POST /v1/chains`:

```json
{
  "chain_id": "genai-example.mcp_tests",
  "chain_config": {
    "package_id": "simple_mcp_chain-0.7.0rc2",
    "chain_module": "simple_mcp_chain.chain",
    "chain_class": "SimpleMcpChain",
    "chain_params": {
      "llm_timeout": 30,
      "gateway_endpoint": "gpt-5.4-mini",
      "opensearch_url": "https://opensearch-genai.stratio-genai:9200"
    }
  }
}
```

Once deployed, Stratio GenAI API automatically registers the embedded MCP server as a tool provider in Stratio GenAI LiteLLM, making its tools available for other applications.

The MCP server name and description are defined in `SimpleMcpChain.chain_params()` with the
`mcp_server_name` and `mcp_server_description` fields of `GenAiChainParams`. `mcp_server_name` is required
when a chain implements `mcp_server()`. You can check the registered server, and the alias used in its URL
(`https://<host>/genai-litellm/<mcp-alias>/mcp`), in the "MCP Servers" page of the LiteLLM UI.

5. Invoke the chain with `POST /v1/chains/simple_mcp_chain/invoke`:

```json
{
  "input": {
    "text": "Hello from GenAI API!"
  }
}
```
