import asyncio
import numpy as np
from multiprocessing import Pool

from genlm_control.potential.base import Potential
from genlm_control.util import LazyWeights

# Notes:
# - multiprocessing does not work well with asyncio:
#   - map_async.get() is blocking, so this is not a true async implementation
#       - the event loop is blocked until the result is returned
#   - the async functions are run synchronously in the worker processes

def mp(potential, init_args, num_workers=2):
    return MultiProcessPotential(potential, init_args, num_workers)

class MultiProcessPotential(Potential):
    """A Potential that adds parallel processing capabilities to any base Potential implementation."""
    def __init__(self, potential_factory, factory_args, num_workers=2):
        self.pool = Pool(
            num_workers,
            initializer=self._init_worker,
            initargs=(potential_factory, factory_args)
        )
        # maybe TODO: use shared memory to pass the weights to the main process
        vocabulary, eos = self.pool.map(self._get_attrs, [None])[0]
        super().__init__(vocabulary, eos)

    @staticmethod
    def _init_worker(factory, args):
        """Initialize the worker process with a potential and an event loop."""
        global _worker_potential, _worker_event_loop
        _worker_potential = factory(*args)
        _worker_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_event_loop)

    @staticmethod
    def _get_attrs(self):
        return _worker_potential.decode, _worker_potential.eos

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
        )

    @staticmethod
    def _worker_prefix(context):
        """Worker process function for computing prefix.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling prefix on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(
            _worker_potential.prefix(context)
        )

    @staticmethod
    def _worker_complete(context):
        """Worker process function for computing complete.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling complete on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(
            _worker_potential.complete(context)
        )

    @staticmethod
    def _worker_score(context):
        """Worker process function for computing score.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling score on the worker's potential instance
        """
        return MultiProcessPotential._run_coroutine(
            _worker_potential.score(context)
        )

    async def logp_next(self, context):
        """Compute p_next for a single context.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling p_next on a worker potential instance
        """
        results = self.pool.map_async(self._worker_logp_next, [context]).get()[0]
        return self.make_lazy_weights(results)

    async def prefix(self, context):
        """Compute prefix for a single context.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling prefix on a worker potential instance
        """
        return self.pool.map_async(self._worker_prefix, [context]).get()[0]

    async def complete(self, context):
        """Compute complete for a single context.

        Args:
            context (List[bytes]): The context to process

        Returns:
            The result of calling complete on a worker potential instance
        """
        return self.pool.map_async(self._worker_complete, [context]).get()[0]

    async def batch_logp_next(self, contexts):
        """Compute p_next for multiple contexts in parallel.

        Args:
            contexts (List[List[bytes]]): List of contexts to process

        Returns:
            (np.array): Results of p_next for each context
        """
        results = self.pool.map_async(self._worker_logp_next, contexts).get()
        return np.array([self.make_lazy_weights(result) for result in results])

    async def batch_score(self, contexts):
        """Compute score for multiple contexts in parallel.

        Args:
            contexts (List[List[bytes]]): List of contexts to process

        Returns:
            (list): Results of score for each context
        """
        results = self.pool.map_async(self._worker_score, contexts).get()
        return np.array(results)

    async def batch_complete(self, contexts):
        """Compute complete for multiple contexts in parallel.

        Args:
            contexts (List[List[bytes]]): List of contexts to process

        Returns:
            (list): Results of complete for each context
        """
        results = self.pool.map_async(self._worker_complete, contexts).get()
        return np.array(results)

    async def batch_prefix(self, contexts):
        """Compute prefix for multiple contexts in parallel.

        Args:
            contexts (List[List[bytes]]): List of contexts to process

        Returns:
            (list): Results of prefix for each context
        """
        results = self.pool.map(self._worker_prefix, contexts)
        return np.array(results)

    def __del__(self):
        """Cleanup method to properly terminate and join the process pool."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None