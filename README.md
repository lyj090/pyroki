# `PyRoki`: Python Robot Kinematics Library

**[Project page](https://pyroki-toolkit.github.io/) &bull;
[arXiv](https://arxiv.org/abs/2505.03728)**

`PyRoki` is a modular, extensible, and cross-platform toolkit for kinematic optimization, all in Python.

Core features include:

- Differentiable robot forward kinematics model from a URDF.
- Automatic generation of robot collision primitives (e.g., capsules).
- Differentiable collision bodies with numpy broadcasting logic.
- Common cost implementations (e.g., end effector pose, self/world-collision, manipulability).
- Arbitrary costs, autodiff or analytical Jacobians.
- Integration with a [Levenberg-Marquardt Solver](https://github.com/brentyi/jaxls) that supports optimization on manifolds (e.g., [lie groups](https://github.com/brentyi/jaxlie)) and hard constraints via an Augmented Lagrangian solver.
- Cross-platform support (CPU, GPU, TPU) via JAX.

Please refer to the [documentation](https://chungmin99.github.io/pyroki/) for more details, features, and usage examples.

---

## Installation

You can install `pyroki` with `pip`, on Python 3.10+:

```
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

## Status

_May 6, 2025_: Initial release

We are preparing and will release by _May 16, 2025_:

- [x] Examples + documentation for hand / humanoid motion retargeting
- [x] Documentation for using manually defined Jacobians
- [x] Support with Python 3.10+

## Limitations

- **Static shapes & JIT overhead**: JAX JIT compilation is triggered on first run and when input shapes change (e.g., number of targets, obstacles). Arrays can be pre-padded to vectorize over inputs with different shapes.
- **No sampling-based planners**: We don't include sampling-based planners (e.g., graphs, trees).
- **Collision performance**: Speed and accuracy comparisons against other robot toolkits such as CuRobo have not been extensively performed, and is likely slower than other toolkits for collision-heavy scenarios.

The following are current implementation limitations that could potentially be addressed in future versions:

- **Joint types**: We only support revolute, continuous, prismatic, and fixed joints. Other URDF joint types are treated as fixed joints.
- **Collision geometry**: We are limited to sphere, capsule, halfspace, and heightmap geometries. Mesh collision is approximated as capsules.
- **Kinematic structures**: We only support kinematic trees; no closed-loop mechanisms or parallel manipulators.

## Citation

This codebase is released with the following preprint.

<table><tr><td>
    Chung Min Kim*, Brent Yi*, Hongsuk Choi, Yi Ma, Ken Goldberg, Angjoo Kanazawa.
    <strong>PyRoki: A Modular Toolkit for Robot Kinematic Optimization</strong>
    arXiV, 2025.
</td></tr>
</table>

<sup>\*</sup><em>Equal Contribution</em>, <em>UC Berkeley</em>.

Please cite PyRoki if you find this work useful for your research:

```
@inproceedings{kim2025pyroki,
  title={PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
  author={Kim*, Chung Min and Yi*, Brent and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  url={https://arxiv.org/abs/2505.03728},
}
```

Thanks!

01_basic_ik.py：单臂基础 IK，拖动一个末端目标位姿，机械臂实时跟随。
02_bimanual_ik.py：双臂 IK（YuMi），两个末端目标同时约束。
03_mobile_ik.py：移动底盘 + 机械臂联合 IK，底盘在平面内移动并允许偏航。
04_ik_with_coll.py：带避障 IK，和地面/球体障碍物保持无碰。
05_ik_with_manipulability.py：IK + 可操作度优化，兼顾到达目标与灵巧性。
06_online_planning.py：在线重规划，持续输出短时域轨迹并随障碍变化更新。
07_trajopt.py：离线轨迹优化（如越过墙体），整段轨迹满足运动与碰撞约束。
08_ik_with_mimic_joints.py：仿真含 mimic joints 的链条，验证 IK 对联动关节支持。
09_hand_retargeting.py：手部动作重定向（MANO → Shadow Hand）基础版。
10_humanoid_retargeting.py：人体关键点动作重定向到 G1 人形机器人基础版。
11_hand_retargeting_fancy.py：手部重定向增强版，加入接触保持等代价项。
12_humanoid_retargeting_fancy.py：人形重定向增强版，加入足底接触、防打滑、世界碰撞等约束。
13_ik_with_locked_joints.py：可锁定关节的 IK，支持动态锁/解锁并查看误差。