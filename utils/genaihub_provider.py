from typing import Dict, Optional, Sequence

import openai
from gen_ai_hub.orchestration.exceptions import OrchestrationError
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.service import OrchestrationService
from retry import retry
from trulens.feedback import LLMProvider

from app.modules.llm import GenAIHubLLM


class GenAIHubProvider(LLMProvider):
    def __init__(self, *args, **kwargs):
        self_kwargs = dict(kwargs)

        super().__init__(**self_kwargs)
        self._orchestration_service = OrchestrationService(api_url=GenAIHubLLM.get_orchestration_deployment_url())

    @retry(
        (openai.RateLimitError, openai.APITimeoutError, OrchestrationError),
        tries=10,
        delay=10,
        backoff=1,
    )
    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs,
    ) -> str:
        temperature = kwargs.pop("temperature", 0)
        if kwargs:
            raise NotImplementedError("kwargs other than temperature not supported")

        if messages is not None:
            msgs = messages
        elif prompt is not None:
            msgs = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        config: OrchestrationConfig = get_orchestration_config(
            model_engine=self.model_engine, temperature=temperature, messages=msgs
        )

        print(f"Calling {self.model_engine} with {(' | '.join([m.get('content') for m in msgs])).replace('\n', ' ')}")
        result = self._orchestration_service.run(config)
        res = result.orchestration_result.choices[0].message.content
        # print(f"Response from {self.model_engine}: {res.replace("\n", " ")}")
        return res


def get_orchestration_config(model_engine, temperature: int, messages: Sequence[dict]) -> OrchestrationConfig:
    llm = LLM(
        name=model_engine,
        version="latest",
        parameters={"temperature": temperature, "max_tokens": 10000},
    )
    template_messages = []
    for message in messages:
        if message["role"] == "system":
            template_messages.append(SystemMessage(message["content"]))
        elif message["role"] == "assistant":
            template_messages.append(AssistantMessage(message["content"]))
        elif message["role"] == "user":
            template_messages.append(UserMessage(message["content"]))

    return OrchestrationConfig(llm=llm, template=Template(messages=template_messages))
