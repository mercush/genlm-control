import asyncio
import numpy as np
from multiprocessing import Pool

from genlm_control.potential.base import Potential


def mp(potential, init_args, num_workers=2):
    return MultiProcessPotential(potential, init_args, num_workers)


class MultiProcessPotential(Potential):
    """A Potential that adds parallel processing capabilities to any base Potential implementation."""

    def __init__(self, potential_factory, factory_args, num_workers=2):
        self.pool = Pool(
            num_workers,
            initializer=self._init_worker,
            initargs=(potential_factory, factory_args),
        )
        # maybe TODO: use shared memory to pass the weights to the main process
        decode = self.pool.map(self._get_decode, [None])[0]
        super().__init__(decode)

    @staticmethod
    def _init_worker(factory, args):
        """Initialize the worker process with a potential and an event loop."""
        global _worker_potential, _worker_event_loop
        _worker_potential = factory(*args)
        _worker_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_event_loop)

    @staticmethod
    def _get_decode(self):
        return _worker_potential.decode

    @staticmethod
    def _run_coroutine(coroutine):
        """Run a coroutine on the worker's event loop."""
        global _worker_event_loop
        return _worker_event_loop.run_until_complete(coroutine)

    @staticmethod
    def _worker_logp_next(context):
        """Worker process function for computing p_next."""
        return MultiProcessPotential._run_coroutine(
            _worker_potential.logp_next(context)
        ).weights

    @staticmethod
    def _worker_prefix(context):
        """Worker process function for computing prefix.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling prefix on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(_worker_potential.prefix(context))

    @staticmethod
    def _worker_complete(context):
        """Worker process function for computing complete.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling complete on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(_worker_potential.complete(context))

    @staticmethod
    def _worker_score(context):
        """Worker process function for computing score.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling score on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(_worker_potential.score(context))

    @staticmethod
    def _worker_logp_next_seq(context, extension):
        return MultiProcessPotential._run_coroutine(
            _worker_potential.logp_next_seq(context, extension)
        )

    async def _run_in_pool(self, func, *args):
        """Run a function in the process pool and return a Future."""
        loop = asyncio.get_running_loop()
        future = asyncio.Future()

        def _callback(result):
            loop.call_soon_threadsafe(future.set_result, result)

        def _error_callback(exc):
            loop.call_soon_threadsafe(future.set_exception, exc)

        self.pool.apply_async(
            func, args, callback=_callback, error_callback=_error_callback
        )

        return await future

    async def logp_next(self, context):
        """Compute p_next for a single context."""
        result = await self._run_in_pool(self._worker_logp_next, context)
        return self.make_lazy_weights(result)

    async def prefix(self, context):
        """Compute prefix for a single context."""
        return await self._run_in_pool(self._worker_prefix, context)

    async def complete(self, context):
        """Compute complete for a single context."""
        return await self._run_in_pool(self._worker_complete, context)

    async def logp_next_seq(self, context, extension):
        return await self._run_in_pool(self._worker_logp_next_seq, context, extension)

    async def batch_logp_next(self, contexts):
        """Compute p_next for multiple contexts in parallel."""
        results = await asyncio.gather(
            *(
                self._run_in_pool(self._worker_logp_next, context)
                for context in contexts
            )
        )
        return [self.make_lazy_weights(result) for result in results]

    async def batch_complete(self, contexts):
        """Compute complete for multiple contexts in parallel."""
        results = await asyncio.gather(
            *(self._run_in_pool(self._worker_complete, context) for context in contexts)
        )
        return np.array(results)

    async def batch_prefix(self, contexts):
        """Compute prefix for multiple contexts in parallel."""
        results = await asyncio.gather(
            *(self._run_in_pool(self._worker_prefix, context) for context in contexts)
        )
        return np.array(results)

    def __del__(self):
        """Cleanup method to properly terminate and join the process pool."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
