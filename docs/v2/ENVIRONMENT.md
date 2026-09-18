# Fleet environment specification

DispatchLab Fleet is a synthetic teaching environment, not a calibrated logistics
system. The original one-vehicle model remains available for learning the basics.

## Decisions and constraints

Two vehicles share one depot and six delivery destinations on a schematic map.
One central dispatcher chooses a joint command every tick. Each vehicle has eight
commands: hold, return to depot, or visit one of six destinations. The flattened
joint action space has 64 entries. Both commands are chosen before either vehicle
moves. Vehicles in transit or completing service can only hold.

Each vehicle carries three parcels, has an 18-unit battery, and charges at three
units per stationary depot tick. The system holds at most ten active orders.
Delivery commands must leave enough energy to return to the hub. The action mask
enforces this reserve before dispatch. Energy is debited once at departure based
on base distance, not again at every travel tick. Congestion increases elapsed
time but not battery consumption in this simplified model.

Loading is automatic: priority first, then deadline and order ID. Vehicles
alternate first access to the queue by tick parity. Returned undelivered parcels
remain onboard. The agent does not choose parcel assignment or loading order.

## Time and travel

A shift lasts 60 abstract ticks. Arrivals stop before tick 44. Base route time is
the ceiling of half the Manhattan distance between node coordinates. Each trip
has a possible extra delay. Normal delay probability is 0.18; during ticks 18–35,
add 0.25. Normal delays add two ticks. Once dispatched, the sampled duration is
known and the remaining duration is observable. Routing is between locations;
the agent does not choose individual road edges.

On arrival, deliveries unload and the vehicle spends the next tick completing
service. Depot charging can occur during that service tick. Loading happens
after service is finished. Both vehicles advance every environment step.

## Orders

Poisson arrivals have mean 0.62 per tick, with two additional initial requests.
Initial requests are also subject to the queue limit. Destinations shift from
the northern to the southern locations halfway through the arrival window.
Deadline slack is uniformly sampled from 10 through 18 ticks. A quarter of orders
have priority 2; the rest have priority 1. Overflow requests are rejected and
count against service metrics and reward. Late orders do not disappear.

## Reward

- Deliver a parcel: +10 × priority.
- Moving vehicle: −0.30 per travel tick.
- Each overdue active parcel: −0.35 × priority per tick, including a tick on
  which it is delivered late.
- Rejected or terminally unfinished parcel: −12 × priority.
- Charging: −0.04 per battery unit added.

Being exactly on the deadline is on time. Rewards are synthetic points. These
weights encode preferences; they are not calibrated revenue, prices, or costs.

## Observation and information separation

The model receives a 164-element vector with current time, known operating
parameters, both vehicle states, and a padded list of revealed orders. The order
list includes destination, carrier, deadline slack, priority, and age. Orders are
sorted deterministically by deadline and ID. The neural encoder supports exactly
this two-vehicle, ten-order design; it is not invariant to fleet size.

Separate random streams generate demand and route-delay fields before the shift.
Policies receive neither future stream. Changing traffic settings with the same
seed leaves demand unchanged. Matching seeds expose identical demand and the
same time/origin/destination delay field to each policy. Policies may experience
different delays because they choose different trips at different times.

## Baselines

Nearest delivery and earliest deadline are reactive dispatch rules. The route
planner enumerates permutations of up to three currently loaded destinations for
each vehicle, checks energy reserve at every stop, and scores expected travel,
service, lateness, and horizon penalties before returning to the hub. It replans
at subsequent decisions. A batch threshold is selected on validation from 1, 2,
or 3 parcels. All baselines use the same action masks and charging rule.

This is a receding-horizon onboard route planner, not a globally optimal dynamic
vehicle-routing solver. It does not optimise loading, joint assignment, or future
unknown requests. It is intentionally stronger than comparing only to random
actions or the closest destination.

## Remaining limitations

No real road network or demand calibration; no pickup points outside the hub;
fixed battery use per base distance; no charger contention, driver breaks,
maintenance, cancellation, weather, or uncertain service times. Known traffic
parameters and sampled-trip duration are exposed immediately. The finite
training budget and simulation assumptions limit the conclusions.
