
#this agent is a PipelineWorker that runs the pipeline.

import os
import sys
import asyncio
import json

from dotenv import load_dotenv
from loguru import logger

DEEPGRAM_STT = os.getenv("DEEPGRAM_STT")
CARTESIA_TTS = os.getenv("CARTESIA_TTS")
OPENAI_LLM = os.getenv("OPENAI_LLM")

#importing packages:--------------------------------------

#workers, pipelines, frames, processors:
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import RunnerArguments

#development runner: for session initialisation
from pipecat.runner.types import RunnerArguments
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

#transport: (using LiveKit)
from pipecat.runner.utils import create_transport
from pipecat.runner.livekit import configure
from pipecat.transports.livekit.transport import (
    LiveKitTransport,
    LiveKitParams
)

#VAD:
from pipecat.audio.vad.silero import SileroVADAnalyzer #local VAD
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
    LLMContextAggregatorPair,
    LLMAssistantAggregatorParams,
)
#turn detection: (smart model - default)
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_start import VADUserTurnStartStrategy #interruption handling

#STT:
from pipecat.services.deepgram.stt import DeepgramSTTService

#LLM:
from pipecat.services.openai.llm import OpenAILLMService

#function calling:
from pipecat.services.llm_service import FunctionCallParams

#TTS:
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.utils.text.pattern_pair_aggregator import PatternPairAggregator, MatchAction
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor

#context management:
from pipecat.frames.frames import (
    LLMMessagesUpdateFrame, #replaces the context box.
    LLMMessagesAppendFrame #adds-on to the context box.
)


#---
load_dotenv(override=True)
logger.remove(0) #deleting default handler
logger.add(sys.stderr, level="DEBUG") #logging destination: error stream; severity level: debug (for errors and failures)
#---

#----------------------------------------------------------------

async def main():
    
#livekit configure:
    url, token, _ = await configure()
#Transport config:
    transport_params = {
        "livekit": 
            lambda: LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            ),
    }
    params = transport_params["livekit"] #getting the params by calling the lamda.

    transport = LiveKitTransport(
        url=url,
        token=token, #auto-generatoed by pipecat
        params = params
    )


#VAD and AI-powered Turn Detection model:
    vad_analyser = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.2, #sensitivity to detection
            start_secs=0.2, #how long a user must speak before speech detected
            stop_secs=0.2, #how long after speech stopped
            min_volume=0.7, #min vol/amplitude required
        )
    )

    stop_strategy = TurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=LocalSmartTurnAnalyzerV3()
    )

    start_strategy = VADUserTurnStartStrategy( #clears the audio and text
        enable_interruptions=True #default
    )

#STT Integration:
    stt = DeepgramSTTService(
        api_key= DEEPGRAM_STT,
        settings=DeepgramSTTService.Settings(
            model="nova2",
            profanity_filter=True,
            diarize=True,
            language="en-US", #or "multi" for multilingual mode
            punctuate=True, #puts punctuation
            interim_results=True #considers interim uttereaces too
        )
    )

#TTS Integration:

    ## lets implement a pattern pair aggregator:
    pattern_aggregator = PatternPairAggregator()
    pattern_aggregator.add_pattern(
        type="<code>", #i dont want it to speak out the code snippets.
        start_pattern="<code>",
        end_pattern="/<code>",
        action= MatchAction.AGGREGATE #makes it an aggregaotor object.
    )
    ## (using LLMTextProcessor for processing the aggregator before it enters the tts integration.)
    llm_text_processor = LLMTextProcessor(
        text_aggregator=pattern_aggregator
    )
    
    
    tts = CartesiaTTSService(
        api_key=CARTESIA_TTS,
        settings=CartesiaTTSService(
            model="sonic-3,5",
            aggregate_sentences=True,
            voice_id="ba2a9c5c-b769-49e3-bc9f-1c2c24bab128", #baron commercial
        ),
        text_aggregation_mode=["code"]
    )
    
#llm integration:
    llm = OpenAILLMService(
    api_key = OPENAI_LLM,
    settings=OpenAILLMService.Settings(
        model="gpt-4o",
        system_instruction="You are helpful voice assisstant." #personality only.
        )
    )
    
    ## llm event-handlers: for monitoring using logger
    @llm.event_handler("on_client_disconnected")
    async def on_completion_timeout(service): #if a service hangs
        
        logger.warning(f"Service {service.model_name} has timed out.")
        
        #inform the user what has happened via TTS.
        await tts.queue_frame([
            TTSSpeakFrame("The client has disconnected.")
        ])
        
        #pipeline (worker) termination:
        await worker.cancel()
        
#pipeline:
    pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator, #context_aggregator.user()
    llm,
    tts,
    transport.output(),
    assisstant_aggregator #context_aggregator.assisstant()
    ])
    ## worker environment:
    worker = PipelineWorker(
        #idle detection and so termination:
        pipeline=pipeline,
        cancel_on_idle_timeout=True,
        idle_timeout_secs=300, #default.
    )

#context management for user and assisstant aggregator:
    ## establishing phase rules via dev messages:
    messages = [ #part of the normal context flow.
        {'role': 'developer', 'content': 'Keep the responses under two minutes.'}
    ]
    context = LLMContext(messages)

    user_aggregator, assisstant_aggregator = LLMContextAggregatorPair(
        context=context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyser,
            user_turn_strategies=True,
        ),
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True #to prevent over-usage of tokens and context width in llms.
        )
    )
    ## task-specific (event and not a tool) context dev messages
    async def transition_to_troubleshooting(
        current_context
    ):
        new_task_messages = [
            {'role': 'developer', 'content': 'TASK: Troubleshooting. Keep responses ultra-short. Tell the user to do unplaug the router and wait for 10 seconds.'}
        ]
        await worker.queue_frames([
            LLMMessagesAppendFrame(messages=new_task_messages, run_llm=True)
        ])


    ## when the bot is supposed to speak first:
    @transport.event_handler("on_client_connected") #when client enters the room
    async def on_client_connected(transport, client):
        logger.info(f"Participant/Client joined: {client['id']}") #logging//
        
        new_message = [
            {'role':'developer', 'content': 'TASK: Introduction. Introduce yourself as an assistant to your manager named Pritish. Ask how you could assist them.'}
        ]
        await worker.queue_frame(
            LLMMessagesAppendFrame([new_message], run_llm=True)
        )


#function tool calling (!): "tools" which the agent can call
    #eg: getting weather:
    async def getting_weather(
        params: FunctionCallParams,
        location: str,
        format: str
    ):
        #tool description (or schema):
        """ 
        Get the current weather.

        Args:
            location: The city and state, e.g. "San Francisco, CA".
            format: The temperature unit to use. Must be either "celsius" or "fahrenheit".
        """
        
        await params.result_callback(
            {"condition": "sunny", "temperature": "75"}
        )

#running the agent:
if __name__ == "__main__":
    asyncio.run(main())
    
    