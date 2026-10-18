# Modeling-Control-of-a-double-pendulum
# Double Pendulum: Swing-Up, Upright Stabilization and Disturbance Rejection

Swing-up and stabilization of a fully actuated double pendulum at the **top vertical** position, using

- **Trajectory optimization** with Hermite–Simpson direct collocation, solved with **IPOPT** (`cyipopt`) and JAX for the model and derivatives
- **TVLQR** (time-varying LQR) to track the optimized swing-up trajectory
- **Infinite-horizon LQR with integral action (LQI)** to hold the pendulum on top and reject disturbances

![Swing-up, hold on top, and disturbance rejection](pendulum_stabilization_disturbance.gif)

The animation shows the swing-up, the hold on top, two short pushes (blue pendulum), and a sustained force from 11 s onward. The orange ghost pendulum uses the same pipeline with plain LQR (no integral action) for comparison.

## What is in this repository

| File | Description |
|---|---|
| `Double_pendulum.ipynb` | The full pipeline: model, trajectory optimization, TVLQR, hold controller, experiments, animation |
| `pendulum_stabilization_disturbance.gif` | Preview of the final animation (the notebook regenerates it) |

## Model

- Two point masses, `m1 = m2 = 1 kg`; link lengths `l1 = l2 = 0.5 m`; `g = 9.81 m/s²`
- State `x = [q1, q2, q1_dot, q2_dot]`, where `q1` is the absolute angle of link 1 and `q2` is the angle of link 2 relative to link 1
- `q = 0` is hanging down; the goal is `x_f = [π, 0, 0, 0]` (upright)
- Both joints are actuated: `u = [u1, u2]` are joint torques, limited to `[-10, 10] N·m`
- Integration: RK4 with `dt = 0.05 s` (zero-order hold on the control)

## Method

### 1. Trajectory optimization (cyipopt)

Hermite–Simpson collocation over `T = 5 s` (`K = 100` intervals). Decision variables are the knot states and controls plus the midpoint states and controls. The cost is the Simpson-rule integral of `|u|²`, **including the midpoint controls** (otherwise the optimizer can hide control effort in the midpoints). The initial guess is a straight line in state space from the hanging state to the upright state. Constraint Jacobians come from JAX (`jacfwd`), the gradient from `grad`, and the Jacobian sparsity pattern is detected automatically. JAX runs in float64.

### 2. TVLQR along the nominal trajectory

The dynamics are linearized along the optimized trajectory, and a backward Riccati pass gives time-varying gains `K_k`. The terminal cost is the infinite-horizon LQR cost-to-go `P∞` at the upright equilibrium, so the end of the trajectory hands over smoothly to the hold controller. The control law is `u_k = u_nom,k + K_k (x - x_nom,k)`.

### 3. Hold on top: infinite-horizon LQR with integral action

After the trajectory ends, the controller switches to an LQR at the upright equilibrium, augmented with an integrator on the two joint-angle errors (LQI). A plain LQR recovers from a push but leaves a permanent offset under a sustained force; the integrator removes that offset. The integrator is frozen while the torque is saturated (anti-windup). `USE_INTEGRAL = False` gives plain LQR.

### Default parameters

| Parameter | Value |
|---|---|
| `Q` | `diag(100, 100, 10, 10)` |
| `R` | `diag(0.01, 0.01)` |
| `Q_INT` (integral weight) | `200` |
| Torque limit | ±10 N·m, applied in the simulation (`CLIP_TORQUE = True`) |

`Q_INT` is the main tuning knob: larger values remove offsets faster but make the controller more sensitive to measurement noise.

## Experiments in the notebook

1. **Measurement noise.** Gaussian noise (σ = 0.001, 0.01, 0.05) on the measured state, with the torque limit active.
2. **Disturbance scenario** (disturbances are extra torques on the joints; a force `F` at the tip is equivalent to the torque `Jᵀ F`):
   - 0–5 s: swing-up with TVLQR, then hold on top
   - 7.0 s: 0.1 s push of +10 N·m on joint 1
   - 9.5 s: 0.1 s push of −6 N·m on joint 2
   - from 11 s: a constant 2 N·m on joint 1
3. **Animation** of the scenario, comparing LQR + integral action with plain LQR.

In the scenario, both hold controllers recover from the two pushes. Under the sustained 2 N·m force, plain LQR settles with a steady offset of about 0.15 rad in `q1` and 0.28 rad in `q2`, while LQR with integral action returns to upright.

## Installation

```bash
conda create -n pendulum python=3.11
conda activate pendulum
conda install -c conda-forge cyipopt   # also installs IPOPT
pip install jax jaxlib numpy scipy matplotlib notebook pillow
```

Then open the notebook and run all cells:

```bash
jupyter notebook Double_pendulum.ipynb
```

The animation is saved as `pendulum_stabilization_disturbance.gif` and also shown inline. Set `INLINE = False` in the last cell to skip the inline player, which makes the notebook file large (several MB).

## Possible extensions

- Robustness tests with parameter perturbations and delays
- Bounds on `q2` or on the joint travel in the trajectory optimization
- A region-of-attraction estimate for the hold controller (for example by sampling or sums-of-squares methods)
- Disturbance observer or adaptive terms for slowly varying loads

## License

Spinos Konstantinos
University of Patras - the project can be used freely and treated as open-source.
