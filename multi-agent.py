# Receptionist and Billing Agent using 

import os
import sys
import json
from dotenv import load_dotenv
from loguru import logger

from __future__ import annotations

import asyncio

from pipecat.workers.llm import (
    LLMWorker, #when you dont need to maintain a context.
    LLMWorkerActivationArgs,
    LLMContextWorker,
    tool
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams

#main:
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

#---
load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
#---

#Entry Point agent:
class ReceptionistAgent(LLMContextWorker): 
    #manages its own conv. context.
    
    @tool
    async def transfer_to_billing(self, billing_inquiry:str):
        "Call this ONLY when the user mentions about paying the bill or anything in regards to the inquiry of invoice."
        
        #logger-
        
        messages = [
            {
                'role': 'developer',
                'content': 
                    f"CONTEXT: The user wants to enquire about the billing"
                    f"Billing Enquiry: {billing_inquiry}"
            }
        ]
        
        return_args = LLMWorkerActivationArgs(
            messages=messages,
            run_llm=True
        )
        await self.activate_worker("billing", deactivate_self=True)
        
        return "Transferring you to the billing agent..."

def building_reception_agent() -> ReceptionistAgent:
    
    llm = OpenAILLMService(
        model="gpt-4o",
        api_key = os.getenv("OPENAI_LLM"),
        settings=OpenAILLMService.Settings(
            system_instruction="You are a Receptionist for PipeCat Corporation. Introduce yourself politely. Be brief and precise. "
        )
    )
    return ReceptionistAgent("receptionist", llm=llm)

class BillingAgent(LLMContextWorker):
    
    @tool #arbitrary for now.
    async def validate_invoice(self, invoice_id: str) -> dict:
        """Looks up an invoice status.
        """
        logger.debug(f"The current billing history length: {len(self.context.messages)}")
        return {"status": "Paid", "amount": "150"} #to make it easier
    
    # AGENT HANDOFF 
    #we need the context of the billing agent to be transfered to the main desk.
    @tool 
    async def transfer_to_receptionist(self, resolution_summary):
        """Call this only when the user's financial queries have been answered."""
        
        logger.info(f"Billing Fare resolved. Proceeding to transfer the agent control back to the receptionist.")
        
        messages = [
            {
                "role": "developer",
                "content": (
                    f"CONTEXT: The user is finished with billing."
                    f"Resolution Summary: {resolution_summary}"
                    f"Acknowledge that their billing issue is sorted out and ask them if there is anything that you can help them with."
                )
            }
        ]
        #worker activation using messages:
        return_args = LLMWorkerActivationArgs(
            messages=messages, #injection
            run_llm=True #indicates that the bot must speak immediately at this instance (injection).
        )
        
        await self.activate_worker(
            worker_name="receptionist".capitalize,
            deactivate_self=True,
            args= return_args
        )
        
        return "Transferring you back to the main desk.."
    
        
def building_billing_agent() -> BillingAgent:
        
    llm = OpenAILLMService(
        api_key = os.getenv("OPENAI_LLM"),
        settings= OpenAILLMService.Settings(
            model="gpt-4o",
            system_instruction="You are the senior accountant specialist. Be formal and precise." #no introductions.
        )
    )
    
    return BillingAgent("billing", llm=llm)

async def main():
    
    #making the pipeline:
    transfer_params = {
        "daily": lambda: 
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            ),
        "livkit": lambda:
            LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            )
    }
     
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_STT"),
        settings=DeepgramSTTService.Settings(
            model="nova2",
            profanity_filter=True,
            interim_results=True,
            punctuate=True,
        )
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_TTS"),
        settings=CartesiaTTSService.Settings(
            model="sonic-3.5"
        )
    )
    
    llm = OpenAILLMService(
        api_key = os.getenv("OPENAI_LLM"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            system_instruction="You are a kind and helpful voice agent."
        )
    )
    
    #MULTI-agent systems use a worker-runner that automatically and cumulatively packages the inbound and outbound transport layers including the cascading of the pipeline. 
    #no need to create the pipelines independently, the runner handles it.
    runner = WorkerRunner()

    #building the workers (or sub-agents)
    runner.add_workers([building_reception_agent(),building_billing_agent()])
    
    #activating a worker on first client connected.
    @runner.event_handler("on_client_connected")
    async def on_client_connected(client):

        
    
        pass    
if __name__ == "__main__":
    pass

