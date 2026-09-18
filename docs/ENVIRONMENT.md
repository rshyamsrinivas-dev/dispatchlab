# How the delivery world works

This is a small synthetic dispatch problem, not a digital twin of a real operator.
One tick is an abstract time unit. No conversion to real minutes is claimed.

## State and actions

One vehicle starts at a depot. It can carry three parcels. Three customer
locations have fixed, symmetric, integer travel times. The system can hold five
active orders, including loaded parcels. Each order has an arrival time,
destination, deadline, and waiting/onboard status. All of this and time remaining
are observed. The vehicle cannot see future orders.

| Action | Feasibility |
|---|---|
| 0: wait | Always allowed; the only action while travelling |
| 1: return to depot | Allowed away from depot, while stationary |
| 2/3/4: North/East/South | Requires an onboard parcel for that destination |

The core environment rejects infeasible actions. All shipped policies use the
mask. The optional Gymnasium adapter instead treats a masked action as a one-tick
wait and subtracts one additional point, so its complete Discrete(5) action space
is defined. Valid-action trajectories are identical in both APIs.

## Event order in one tick

1. Accept the action; start a trip if requested.
2. Advance travel by one tick and charge 0.35 points if moving.
3. Advance the clock.
4. If the vehicle has arrived, deliver every onboard parcel for that location.
5. Charge 0.7 points per overdue active order, including a late delivery on this
   tick. Being exactly on the deadline is on time.
6. Reveal any new request scheduled for this time. If five orders are already
   active, reject it and charge eight points.
7. At the depot, load available capacity by earliest deadline, then order ID.
8. At tick 48, terminate and charge eight points per unfinished order. No extra
   travel beyond the shift is allowed.

Loading and unloading have zero service time. Returned, undelivered cargo stays
on the vehicle. Loading selection is fixed; the agent does not optimise it.

## Demand and information boundaries

At reset, the simulator generates the entire request stream using a dedicated
seed. A request occurs at tick zero; subsequent ticks before 36 independently
have probability 0.38 of one request. Deadlines are arrival plus 9–16 ticks.
Destination probabilities are North/East/South = 0.60/0.25/0.15 before tick 18,
then 0.15/0.25/0.60. Policies know only requests revealed so far.

The private schedule is not part of observations or info. Policies are functions
of observation and action mask, never receive the environment object, and cannot
inspect its private schedule. Identical scenario seeds produce identical offered
requests for every policy even if their rejection counts differ.

## Reward

Per tick: 10 × deliveries − 0.35 × travel ticks − 0.7 × overdue orders
− 8 × rejected orders. At the terminal tick, subtract 8 × unfinished orders.
These are chosen teaching weights, not prices, savings, or estimated business value.

Reward includes rejection and unfinished penalties to reduce the incentive to
ignore difficult work. The choices still encode trade-offs: real applications
would need calibrated costs, stakeholder review, and constraint checks.

## Finite horizon

The 48-tick shift is part of the problem and remaining time is observed.
It ends with `terminated=True`, `truncated=False`; the learner does not bootstrap
from this terminal state. Trips selected near closing time can remain unfinished.
Their remaining parcels are counted and penalised.

## What the demo shows

The HTML replays recorded outputs from the actual Python simulator. It does not
reimplement the simulator in JavaScript or train a model in the browser. The model
is frozen. Changing the scenario or shift selects another recorded trajectory.

## Limitations

Small fixed map; one vehicle; synthetic independent arrivals with a known time
pattern; deterministic travel; no road network, traffic, battery, cancellations,
driver breaks, prioritised customers, or service times. Only one arrival per tick.
Orders may be delivered late. These omissions make the first project learnable,
but sharply limit any inference about real logistics performance.
