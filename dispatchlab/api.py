"""Local demonstration API. Run with: uvicorn dispatchlab.api:app --port 8000.

No authentication, persistence, external services, or deployment is configured.
Keep this teaching server bound to localhost (the default).
"""

from functools import lru_cache
from pathlib import Path
import json
from dataclasses import replace
from typing import Literal
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .fleet import FleetConfig
from .fleet_policies import fleet_heuristic
from .fleet_experiment import rollout
from .deep_agent import DoubleDQN

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="DispatchLab Fleet",
    version="0.2.0",
    description="Synthetic, energy-constrained fleet dispatch. Local demonstration only.",
)


class SimulationRequest(BaseModel):
    seed: int = Field(default=2000001, ge=0, le=2**31 - 1)
    scenario: Literal[
        "standard", "demand_surge", "traffic_disruption", "energy_constraint"
    ] = "standard"
    policy: Literal["ddqn", "planner", "deadline", "nearest"] = "ddqn"


@lru_cache(maxsize=1)
def model():
    return DoubleDQN.load(ROOT / "advanced/models/ddqn.pt")


@app.get("/")
def home():
    return FileResponse(ROOT / "advanced/demo/index.html")


@app.get("/health")
def health():
    return dict(
        status="ok",
        version="0.2.0",
        model_available=(ROOT / "advanced/models/ddqn.pt").is_file(),
        synthetic=True,
    )


@app.get("/benchmark")
def benchmark():
    return json.loads((ROOT / "advanced/results/summary.json").read_text())


@app.post("/simulate")
def simulate(request: SimulationRequest):
    cfg = FleetConfig()
    configs = {
        "standard": cfg,
        "demand_surge": replace(cfg, demand_rate=0.95),
        "traffic_disruption": replace(cfg, traffic_probability=0.65, traffic_extra=3),
        "energy_constraint": replace(cfg, battery=12, charge_rate=2),
    }
    manifest = json.loads((ROOT / "advanced/results/training.json").read_text())
    agent = model() if request.policy == "ddqn" else None
    policy = (
        agent.act
        if agent
        else lambda o, m: fleet_heuristic(
            o,
            m,
            {"planner": "planner", "deadline": "deadline", "nearest": "nearest"}[
                request.policy
            ],
            manifest["planner_batch"] if request.policy == "planner" else 1,
        )
    )
    metrics, frames = rollout(
        configs[request.scenario], request.seed, policy, True, agent
    )
    return dict(
        seed=request.seed,
        scenario=request.scenario,
        policy=request.policy,
        metrics=metrics,
        frames=frames,
    )
