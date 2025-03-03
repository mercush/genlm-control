import numpy as np
from genlm_control.potential import Potential
from hypothesis import strategies as st


class MockPotential(Potential):
    def __init__(self, vocab, next_token_logws):
        self.next_token_logws = np.array(next_token_logws)
        super().__init__(vocab)

    def _logw(self, context):
        return sum([self.next_token_logws[self.lookup[i]] for i in context])

    async def prefix(self, context):
        return self._logw(context)

    async def complete(self, context):
        return self._logw(context) + self.next_token_logws[-1]

    async def logw_next(self, context):
        return self.make_lazy_weights(self.next_token_logws)


@st.composite
def mock_vocab(draw):
    item_strategy = draw(
        st.sampled_from(
            (
                st.text(min_size=1),
                st.binary(min_size=1),
            )
        )
    )

    # Sample vocabulary of iterables.
    vocab = draw(st.lists(item_strategy, min_size=1, max_size=10, unique=True))
    return vocab


@st.composite
def mock_vocab_and_ws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    return vocab, ws


@st.composite
def mock_params(draw, max_w=1e3):
    iter_vocab, iter_next_token_ws = draw(mock_vocab_and_ws())

    # Sample context from iter_vocab
    context = draw(st.lists(st.sampled_from(iter_vocab), min_size=0, max_size=10))

    return (iter_vocab, iter_next_token_ws, context)


@st.composite
def iter_item_params(draw, max_iter_w=1e3, max_item_w=1e3):
    iter_vocab, iter_next_token_ws, context = draw(mock_params(max_iter_w))

    item_vocab = set()
    for items in iter_vocab:
        item_vocab.update(items)
    item_vocab = list(item_vocab)

    # Sample weights over item vocabulary and EOS.
    item_next_token_ws = draw(
        st.lists(
            st.floats(1e-5, max_item_w),
            min_size=len(item_vocab) + 1,
            max_size=len(item_vocab) + 1,
        )
    )

    return (iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context)


class WeightedSet(Potential):
    def __init__(self, sequences, weights):
        self.complete_logws = {
            tuple(seq): np.log(w) if w != 0 else float("-inf")
            for seq, w in zip(sequences, weights)
        }

        prefix_ws = {}
        for seq, w in zip(sequences, weights):
            for i in range(0, len(seq) + 1):
                prefix = tuple(seq[:i])
                if prefix not in prefix_ws:
                    prefix_ws[prefix] = 0.0
                prefix_ws[prefix] += w

        self.prefix_log_ws = {
            prefix: np.log(w) if w != 0 else float("-inf")
            for prefix, w in prefix_ws.items()
        }
        total_weight = sum(weights)
        assert np.isclose(
            self.prefix_log_ws[()],
            np.log(total_weight) if total_weight != 0 else float("-inf")
        )

        super().__init__(list(set(t for seq in sequences for t in seq)))

    async def complete(self, context):
        return self.complete_logws.get(tuple(context), float("-inf"))

    async def prefix(self, context):
        return self.prefix_log_ws.get(tuple(context), float("-inf"))


@st.composite
def weighted_sequence(draw, max_seq_len=5):
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight = draw(st.floats(min_value=1e-3, max_value=1e3))
    return sequence, weight


@st.composite
def double_weighted_sequence(draw, max_seq_len=5):
    # We use the second weight as the weight assigned to the sequence
    # by the critic.
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight1 = draw(st.floats(min_value=1e-3, max_value=1e3))
    weight2 = draw(st.floats(min_value=0, max_value=1e3))
    return sequence, weight1, weight2


@st.composite
def weighted_set(draw, item_sampler, max_seq_len=5, max_size=5):
    return draw(
        st.lists(
            item_sampler(max_seq_len),
            min_size=1,
            max_size=max_size,
            unique_by=lambda x: x[0],
        )
    )
