from hfppl import SubModel


class UnitSampler(SubModel):
    def __init__(self, target, unit_type):
        super().__init__()
        self.target = target
        self.unit_type = unit_type

    async def forward(self, context):
        raise NotImplementedError

    def immutable_properties(self):
        return set(["target", "unit_type"])


class DirectTokenSampler(UnitSampler):
    def __init__(self, potential):
        super().__init__(potential, potential.token_type)
        self.potential = potential

    async def forward(self, context):
        logws = await self.potential.logw_next(context)
        token = logws.normalize().sample()
        self.score(logws.sum())
        return token

    def immutable_properties(self):
        return super().immutable_properties() | set(["potential"])


class SetTokenSampler(UnitSampler):
    def __init__(self, set_sampler):
        super().__init__(set_sampler.target, set_sampler.token_type)
        self.set_sampler = set_sampler

    async def forward(self, context):
        logws = await self.set_sampler.sample_set(context)
        token = logws.normalize().sample()
        self.score(logws.sum())
        return token

    def immutable_properties(self):
        return super().immutable_properties() | set(["set_sampler"])
