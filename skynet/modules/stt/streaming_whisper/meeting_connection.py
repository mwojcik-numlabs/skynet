from itertools import chain
from typing import List

from faster_whisper.tokenizer import Tokenizer
from starlette.websockets import WebSocket

from skynet.env import whisper_max_finals_in_initial_prompt as max_finals
from skynet.logs import get_logger
from skynet.modules.stt.streaming_whisper.cfg import model
from skynet.modules.stt.streaming_whisper.chunk import Chunk
from skynet.modules.stt.streaming_whisper.state import State
from skynet.modules.stt.streaming_whisper.utils import utils

log = get_logger(__name__)


class MeetingConnection:
    participants: dict[str, State] = {}
    tokenizers: dict[str, Tokenizer]
    previous_transcription_tokens: dict[str, List[int]]
    previous_transcription_store: dict[str, List[List[int]]]
    total_finals: int
    total_interims: int
    total_audio_received_s: float
    meeting_id: str
    ws: WebSocket
    connected: True

    def __init__(self, ws: WebSocket, meeting_id: str):
        self.participants = {}
        self.ws = ws
        self.meeting_id = meeting_id
        self.tokenizers = {}
        self.previous_transcription_tokens = {}
        self.previous_transcription_store = {}
        self.total_finals = 0
        self.total_interims = 0
        self.total_audio_received_s = 0
        self.connected = True

    def get_tokenizer(self, language: str) -> Tokenizer:
        """Get existing tokenizer for language or create a new one."""
        if language not in self.tokenizers:
            log.debug(f"Creating tokenizer for language: {language}")
            self.tokenizers[language] = Tokenizer(
                model.hf_tokenizer,
                multilingual=False,
                task="transcribe",
                language=language,
            )
            self.previous_transcription_tokens[language] = []
            self.previous_transcription_store[language] = []
        return self.tokenizers[language]

    async def update_connection_summary_stats(self, payloads):
        for payload in payloads:
            if payload.type == "final":
                self.total_finals += 1
            else:
                self.total_interims += 1

    async def update_initial_prompt(
        self, previous_payloads: list[utils.TranscriptionResponse], language: str
    ):
        tokenizer = self.get_tokenizer(language)
        for payload in previous_payloads:
            if payload.type == "final" and not any(
                prompt in payload.text for prompt in utils.black_listed_prompts
            ):
                self.previous_transcription_store[language].append(
                    tokenizer.encode(f" {payload.text.strip()}")
                )
                if len(self.previous_transcription_store[language]) > max_finals:
                    self.previous_transcription_store[language].pop(0)
                # flatten the list of lists
                self.previous_transcription_tokens[language] = list(
                    chain.from_iterable(self.previous_transcription_store[language])
                )

    async def process(
        self, chunk: bytes, chunk_timestamp: int
    ) -> List[utils.TranscriptionResponse] | None:
        a_chunk = Chunk(chunk, chunk_timestamp)
        self.total_audio_received_s += a_chunk.duration

        if a_chunk.participant_id not in self.participants:
            log.debug(
                f"The participant {a_chunk.participant_id} is not in the participants list, creating a new state."
            )
            self.participants[a_chunk.participant_id] = State(a_chunk.participant_id)

        participant = self.participants[a_chunk.participant_id]

        # Get previous tokens for participant's last detected language (empty list if new)
        previous_tokens = self.previous_transcription_tokens.get(
            participant.last_language, []
        )

        payloads = await participant.process(a_chunk, previous_tokens)
        if payloads:
            # Use the detected language from transcription result
            detected_language = participant.last_language
            self.get_tokenizer(detected_language)  # Ensure tokenizer exists
            await self.update_connection_summary_stats(payloads)
            await self.update_initial_prompt(payloads, detected_language)
        return payloads

    async def force_transcription(self, participant_id: str):
        if participant_id in self.participants:
            participant = self.participants[participant_id]
            language = participant.last_language
            previous_tokens = self.previous_transcription_tokens.get(language, [])
            payloads = await participant.force_transcription(previous_tokens)
            if payloads:
                await self.update_connection_summary_stats(payloads)
                await self.update_initial_prompt(payloads, language)
            return payloads
        return None

    def disconnect(self):
        self.connected = False

    async def close(self):
        await self.ws.close()
        self.disconnect()
