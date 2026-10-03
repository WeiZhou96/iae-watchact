"""Preflight caps, logical cache charging, and complete usage reporting."""
from __future__ import annotations
from dataclasses import dataclass,asdict,field

class BudgetExceeded(RuntimeError):pass

@dataclass
class Cost:
    input_tokens:int=0  # includes visual tokens, where backend reports that convention
    visual_tokens:int=0
    output_tokens:int=0
    frame_exposures:int=0
    calls:int=1
    exact:bool=True

    def __post_init__(self):
        for key in ['input_tokens','visual_tokens','output_tokens','frame_exposures','calls']:
            if not isinstance(getattr(self,key),int) or getattr(self,key)<0:
                raise ValueError(f'{key} must be a nonnegative integer')

@dataclass
class Limits:
    calls:int=24
    input_tokens:int=160000
    visual_tokens:int=120000
    output_tokens:int=24000
    frame_exposures:int=1024
    unique_frames:int=256

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            if not isinstance(getattr(self,key),int) or getattr(self,key)<0:
                raise ValueError(f'{key} must be a nonnegative integer')

@dataclass
class Ledger:
    limits:Limits=field(default_factory=Limits)
    used:Cost=field(default_factory=lambda:Cost(calls=0))
    seen_frames:set=field(default_factory=set)
    records:list=field(default_factory=list)

    def can(self,cost:Cost,frame_keys=(),reserve:Cost|None=None)->bool:
        reserve=reserve or Cost(calls=0)
        for key in ['calls','input_tokens','visual_tokens','output_tokens','frame_exposures']:
            if getattr(self.used,key)+getattr(cost,key)+getattr(reserve,key)>getattr(self.limits,key):return False
        return len(self.seen_frames|set(frame_keys))<=self.limits.unique_frames

    def preflight(self,cost:Cost,frame_keys=(),reserve:Cost|None=None):
        if not self.can(cost,frame_keys,reserve):raise BudgetExceeded('Per-example budget exhausted; no unmetered retries')

    def charge(self,cost:Cost,frame_keys=(),record=None):
        for key in ['calls','input_tokens','visual_tokens','output_tokens','frame_exposures']:
            setattr(self.used,key,getattr(self.used,key)+getattr(cost,key))
        self.used.exact=self.used.exact and cost.exact
        self.seen_frames.update(frame_keys)
        self.records.append({**(record or {}),'cost':asdict(cost)})
        if not self.can(Cost(calls=0)):
            raise BudgetExceeded('Backend actual usage exceeded preflight cap; record retained, example marked failed')

    def report(self)->dict:
        return {'limits':asdict(self.limits),'used':asdict(self.used),'unique_frames':len(self.seen_frames),
                'token_accounting':'exact backend tokenizer' if self.used.exact else 'contains estimates; not exact-token matched',
                'records':self.records}
