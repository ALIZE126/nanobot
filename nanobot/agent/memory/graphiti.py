import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti_core.llm_client import OpenAIClient
from loguru import logger
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

# Initialize Azure OpenAI client using the standard OpenAI client
# with Azure's v1 API endpoint
ke_client = AsyncOpenAI(
    base_url="http://chatgpt.ke.com/v1",
    api_key="11622f96-b5d0-4e0b-bf10-3841939044b6",
)

silicon_flow_client = AsyncOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-thpzlrurzxhkbhivokwdjzhocfclinceoimgsktadaxlkqfv",
)

# 1. 创建一个补丁类修复返回值数量不对的问题
class FixedAzureOpenAILLMClient(AzureOpenAILLMClient):
    # 1. 这个方法在底层是异步的，保持 async/await
    async def _generate_response(self, *args, **kwargs):
        result = await super()._generate_response(*args, **kwargs)
        if not isinstance(result, tuple):
            return result, 0, 0
        if len(result) == 1:
            return result[0], 0, 0
        return result

    # 2. 这个方法在底层是同步的，去掉 async 和 await
    def _handle_structured_response(self, *args, **kwargs):
        result = super()._handle_structured_response(*args, **kwargs)
        if not isinstance(result, tuple):
            return result, 0, 0
        if len(result) == 1:
            return result[0], 0, 0
        return result

# Create LLM and Embedder clients
llm_client = FixedAzureOpenAILLMClient(
    azure_client=ke_client,
    config=LLMConfig(model="deepseek-v3.1", small_model="deepseek-v3.1")  # Your Azure deployment name
)
embedder_client = AzureOpenAIEmbedderClient(
    azure_client=silicon_flow_client,
    model="Qwen/Qwen3-Embedding-4B"  # Your Azure embedding deployment name
)

cross_encoder_client = OpenAIRerankerClient(
    client=ke_client,
    config=LLMConfig(model="MiniMax-M1", small_model="MiniMax-M1")  # Your Azure embedding deployment name
)
from graphiti_core.driver.falkordb_driver import FalkorDriver
driver = FalkorDriver(
    host="localhost",
    port=6380,
    database="my_custom_graph"  # Custom database name
)


# Initialize Graphiti with Azure OpenAI clients
graphiti = Graphiti(
    graph_driver=driver,
    llm_client=llm_client,
    embedder=embedder_client,
    cross_encoder=cross_encoder_client,
)


def edge_to_dict(edge: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(edge, "uuid", "")),
        "memory": getattr(edge, "fact", str(edge)),
        "valid_at": str(getattr(edge, "valid_at", "")),
        "invalid_at": str(getattr(edge, "invalid_at", "")),
    }

def messages_to_episode_body(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = m.get("content", "")
        if not content:
            continue
        role = m.get("role", "user").capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# Now you can use Graphiti with Azure OpenAI
class GraphitiMemoryStore:
    def __init__(
        self,
        workspace: Path,
        graphiti = None,
    ):
        # Store raw config; Graphiti instance is created lazily on the first
        # async call so that its internal connection pools are always bound to
        # the running event loop (avoids "Future attached to a different loop").
        self._graphiti: Any = graphiti

    async def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str = "test_user",
        **kwargs: Any,
    ) -> Any:
        """Add a conversation episode to the knowledge graph."""
        body = messages_to_episode_body(messages)
        if not body:
            return {}
        try:
            from graphiti_core.nodes import EpisodeType
            result = await self._graphiti.add_episode(
                name=f"Conversation ({user_id})",
                episode_body=body,
                source_description="nanobot conversation",
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.message,
                group_id=user_id,
            )
            logger.info("add success {}", result)
            return result
        except Exception:
            logger.exception("Graphiti add_episode failed")
            raise

    async def search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Hybrid semantic + keyword search over the knowledge graph."""
        try:
            edges = await self._graphiti.search(
                query=query,
                group_ids=[user_id],
                num_results=limit,
            )
            search_result=[edge_to_dict(e) for e in edges]
            logger.info("search success {}", search_result)
            return search_result
        except Exception:
            logger.exception("Graphiti search failed")
            return []

async def main() -> None:
    g=GraphitiMemoryStore(
        workspace = Path("~/workspace").expanduser(),
        graphiti = graphiti
    )
    messages = [
        {"role": "user", "content": "我叫李明，是一名软件工程师"},
        {"role": "assistant", "content": "你好李明，很高兴认识你！"},
        {"role": "user", "content": "我喜欢用Python做数据分析"},
    ]
    await g.add(messages)
    # await g.search("李明的职业")
if __name__ == "__main__":
    asyncio.run(main())