try:
    import torch
    import asyncio
    import logging
    from queue import Empty, Queue

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

except:
    pass

    from ray import serve
    from typing import List, Dict, Any

logger = logging.getLogger("ray.serve")

class RawStreamer:
    def _init_(self, timeout: float = None):
        self.q = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, values):
        self.q.put(values)

    def end(self):
        self.q.put(self.stop_signal)

    def _iter_(self):
        return self

    def _next_(self):
        result = self.q.get(timeout=self.timeout)
        if result == self.stop_signal:
            raise StopIteration()
        else:
            return result

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class Batchbot:
    def _init_(self, model_id: str):
        self.loop = asyncio.get_running_loop()

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=False)
        self.device_map = [
        # Assign initial layers to GPU 0
        ('model.embed_tokens', 0),
        ('model.layers.0', 0),
        ('model.layers.1', 0),
        ('model.layers.2', 0),
        ('model.layers.3', 0),
        ('model.layers.4', 0),
        ('model.layers.5', 0),
        ('model.layers.6', 0),
        ('model.layers.7', 0),
        ('model.layers.8', 0),
        ('model.layers.9', 0),
        ('model.layers.10', 0),
        ('model.layers.11', 0),
        ('model.layers.12', 0),
        ('model.layers.13', 0),
        ('model.layers.14', 0),
        ('model.layers.15', 0),
        ('model.layers.16', 0),
        ('model.layers.17', 0),
        ('model.layers.18', 0),
        ('model.layers.19', 0),
        ('model.layers.20', 0),
        ('model.layers.21', 0),
        ('model.layers.22', 0),
        ('model.layers.23', 0),
        ('model.layers.24', 0),
        ('model.layers.25', 0),
        ('model.layers.26', 0),
        ('model.layers.27', 0),
        ('model.layers.28', 0),
        ('model.layers.29', 0),


    


        # Assign middle layers to GPU 1
        ('model.layers.30', 1),
        ('model.layers.31', 1),
        ('model.layers.32', 1),
        ('model.layers.33', 1),
        ('model.layers.34', 1),
        ('model.layers.35', 1),
        ('model.layers.36', 1),
        ('model.layers.37', 1),
        ('model.layers.38', 1),
        ('model.layers.39', 1),
        ('model.layers.40', 1),
        ('model.layers.41', 1),
        ('model.layers.42', 1),
        ('model.layers.43', 1),
        ('model.layers.44', 1),
        ('model.layers.45', 1),
        ('model.layers.46', 1),
        ('model.layers.47', 1),
        ('model.layers.48', 1),
        ('model.layers.49', 1),
        ('model.layers.50', 1),
        ('model.layers.51', 1),
        ('model.layers.52', 1),
        ('model.layers.53', 1),
        ('model.layers.54', 1),
        ('model.layers.55', 1),
        ('model.layers.56', 1),
        ('model.layers.57', 1),
        ('model.layers.58', 1),
        ('model.layers.59', 1),
        ('model.layers.60', 1),
        ('model.layers.61', 1),
        ('model.layers.62', 1),
        ('model.layers.63', 1),
        ('model.layers.64', 1),
        ('model.layers.65', 1),

        # Assign final layers to GPU 2
        ('model.layers.66', 2),
        ('model.layers.67', 2),
        ('model.layers.68', 2),
        ('model.layers.69', 2),
        ('model.layers.70', 2),
        ('model.layers.71', 2),
        ('model.layers.72', 2),
        ('model.layers.73', 2),
        ('model.layers.74', 2),
        ('model.layers.75', 2),
        ('model.layers.76', 2),
        ('model.layers.77', 2),
        ('model.layers.78', 2),
        ('model.layers.79', 2),
        ('model.norm', 2),
        ('lm_head', 2)
        ]

        # Convert to dictionary
        self.device_map_dict = {
            key: value for key, value in self.device_map
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, torch_dtype="auto",device_map=self.device_map_dict, low_cpu_mem_usage = True)


    @fastapi_app.post("/")
    async def handle_request(self, prompt: str) -> StreamingResponse:
        logger.info(f'Got prompt: "{prompt}"')
        return StreamingResponse(self.run_model(prompt), media_type="text/plain")

    @serve.batch(max_batch_size=2, batch_wait_timeout_s=15)
    async def run_model(self, prompts: List[str]):
        streamer = RawStreamer()
        self.loop.run_in_executor(None, self.generate_text, prompts, streamer)
        on_prompt_tokens = True
        async for decoded_token_batch in self.consume_streamer(streamer):
            # The first batch of tokens contains the prompts, so we skip it.
            if not on_prompt_tokens:
                logger.info(f"Yielding decoded_token_batch: {decoded_token_batch}")
                yield decoded_token_batch
            else:
                logger.info(f"Skipped prompts: {decoded_token_batch}")
                on_prompt_tokens = False

    def generate_text(self, prompts: str, streamer: RawStreamer):
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000)

    async def consume_streamer(self, streamer: RawStreamer):
        while True:
            try:
                for token_batch in streamer:
                    decoded_tokens = []
                    for token in token_batch:
                        decoded_tokens.append(
                            self.tokenizer.decode(token, skip_special_tokens=True)
                        )
                    logger.info(f"Yielding decoded tokens: {decoded_tokens}")
                    yield decoded_tokens
                break
            except Empty:
                await asyncio.sleep(0.001)

app = Batchbot.bind("meta-llama/Meta-Llama-3-70B-Instruct")
