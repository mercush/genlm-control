import asyncio
import logging
from hfppl import Model

from genlm_control.constant import EOS


class SequenceSampler(Model):
    def __init__(self, max_tokens, generator, critic=None, sync_fn=None):
        super().__init__()
        self.context = []
        self.proposal = Proposal(generator)
        self.max_tokens = max_tokens

        if critic:
            self.critic = Critic(critic)
        else:
            self.critic = None

        self.sync_fn = sync_fn or (lambda x: True)

    async def step(self):
        await self.proposal(self)

        while not self.sync_fn(self.context):
            await self.proposal(self)

        if self.critic:
            self.untwist()
            await self.critic(self)

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.context[-1] is EOS:
            self.finish()
            return

    def immutable_properties(self):
        return set(["proposal", "twist"])

    async def cleanup(self):
        await self.proposal.cleanup()
        if self.critic:
            await self.critic.cleanup()


class BatchSubModel:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start()

    def start(self):
        if not self.task:
            self.task = asyncio.create_task(self._background_loop())

    async def __call__(self, model):
        return await self.forward(model)

    async def forward(self, model):
        if not self.task:
            self.start()

        future = asyncio.Future()
        await self.queue.put((model, future))
        return await future

    async def batch_forward(self):
        raise NotImplementedError("Subclasses must implement batch_forward")

    async def _background_loop(self):
        while True:
            try:
                models = []
                futures = []

                model, future = await self.queue.get()
                models.append(model)
                futures.append(future)

                while not self.queue.empty():
                    model, future = await self.queue.get()
                    models.append(model)
                    futures.append(future)

                self.logger.debug(f"Processing {len(models)} requests")

                await self.batch_forward(models)

                for future in futures:
                    future.set_result(None)

            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                raise

    async def cleanup(self):
        """Async cleanup - preferred method"""
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def destroy(self):
        if self.task:
            self.task.cancel()
            self.task = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class Proposal(BatchSubModel):
    def __init__(self, p):
        self.p = p
        super().__init__()

    async def batch_forward(self, models):
        contexts = [model.context for model in models]
        tokens, ws, _ = await self.p.batch_sample_token(contexts)
        assert len(tokens) == len(models), f"{len(tokens)} != {len(models)}"
        for token, log_w, model in zip(tokens, ws, models):
            model.context.append(token)
            model.score(log_w)


class Critic(BatchSubModel):
    def __init__(self, p):
        self.p = p
        super().__init__()

    async def batch_forward(self, models):
        contexts = [model.context for model in models]
        amts = await self.p.batch_score(contexts)
        assert len(amts) == len(models), f"{len(amts)} != {len(models)}"
        for amt, model in zip(amts, models):
            model.twist(amt)
