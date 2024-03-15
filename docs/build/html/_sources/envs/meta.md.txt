# Metaworld

<div class='center'>
    <img src="../_static/imgs/env_gifs/Metaworld.gif" style="margin: 5%; width: 45%;">
    <p>Metaworld Dial Turn Task (metaworld/dial-turn-v2)</p>
</div>
<br>

[Metaworld](https://meta-world.github.io/) is an open-source simulated benchmark designed to advance meta-reinforcement learning and multi-task learning, comprising 50 diverse robotic manipulation tasks. The benchmark features a universal tabletop environment equipped with a simulated Sawyer arm and a variety of everyday objects. This shared environment is pivotal for reusing structured learning and efficiently acquiring related tasks.

## Step-Based Environments

`fancy_gym` makes all metaworld ML1 tasks avaible via the standard gym interface.

| Name                                     | Description                                                                           | Horizon | Action Dimension | Observation Dimension | Context Dimension |
| ---------------------------------------- | ------------------------------------------------------------------------------------- | ------- | ---------------- | --------------------- | ----------------- |
| `metaworld/assembly-v2`                  | A task where the robot must assemble components.                                      | 150     | 4                | 39                    | 6                 |
| `metaworld/basketball-v2`                | A task where the robot must play a game of basketball.                                | 150     | 4                | 39                    | 6                 |
| `metaworld/bin-picking-v2`               | A task involving the robot picking objects from a bin.                                | 150     | 4                | 39                    | 6                 |
| `metaworld/box-close-v2`                 | A task requiring the robot to close a box.                                            | 150     | 4                | 39                    | 6                 |
| `metaworld/button-press-topdown-v2`      | A task where the robot must press a button from a top-down perspective.               | 150     | 4                | 39                    | 6                 |
| `metaworld/button-press-topdown-wall-v2` | A task involving the robot pressing a button with a wall from a top-down perspective. | 150     | 4                | 39                    | 6                 |
| `metaworld/button-press-v2`              | A task where the robot must press a button.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/button-press-wall-v2`         | A task involving the robot pressing a button with a wall.                             | 150     | 4                | 39                    | 6                 |
| `metaworld/coffee-button-v2`             | A task where the robot must press a button on a coffee machine.                       | 150     | 4                | 39                    | 6                 |
| `metaworld/coffee-pull-v2`               | A task involving the robot pulling a lever on a coffee machine.                       | 150     | 4                | 39                    | 6                 |
| `metaworld/coffee-push-v2`               | A task involving the robot pushing a component on a coffee machine.                   | 150     | 4                | 39                    | 6                 |
| `metaworld/dial-turn-v2`                 | A task where the robot must turn a dial.                                              | 150     | 4                | 39                    | 6                 |
| `metaworld/disassemble-v2`               | A task requiring the robot to disassemble an object.                                  | 150     | 4                | 39                    | 6                 |
| `metaworld/door-close-v2`                | A task where the robot must close a door.                                             | 150     | 4                | 39                    | 6                 |
| `metaworld/door-lock-v2`                 | A task involving the robot locking a door.                                            | 150     | 4                | 39                    | 6                 |
| `metaworld/door-open-v2`                 | A task where the robot must open a door.                                              | 150     | 4                | 39                    | 6                 |
| `metaworld/door-unlock-v2`               | A task involving the robot unlocking a door.                                          | 150     | 4                | 39                    | 6                 |
| `metaworld/hand-insert-v2`               | A task requiring the robot to insert a hand into an object.                           | 150     | 4                | 39                    | 6                 |
| `metaworld/drawer-close-v2`              | A task where the robot must close a drawer.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/drawer-open-v2`               | A task involving the robot opening a drawer.                                          | 150     | 4                | 39                    | 6                 |
| `metaworld/faucet-open-v2`               | A task requiring the robot to open a faucet.                                          | 150     | 4                | 39                    | 6                 |
| `metaworld/faucet-close-v2`              | A task where the robot must close a faucet.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/hammer-v2`                    | A task where the robot must use a hammer.                                             | 150     | 4                | 39                    | 6                 |
| `metaworld/handle-press-side-v2`         | A task involving the robot pressing a handle from the side.                           | 150     | 4                | 39                    | 6                 |
| `metaworld/handle-press-v2`              | A task where the robot must press a handle.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/handle-pull-side-v2`          | A task requiring the robot to pull a handle from the side.                            | 150     | 4                | 39                    | 6                 |
| `metaworld/handle-pull-v2`               | A task where the robot must pull a handle.                                            | 150     | 4                | 39                    | 6                 |
| `metaworld/lever-pull-v2`                | A task involving the robot pulling a lever.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/peg-insert-side-v2`           | A task requiring the robot to insert a peg from the side.                             | 150     | 4                | 39                    | 6                 |
| `metaworld/pick-place-wall-v2`           | A task involving the robot picking and placing an object with a wall.                 | 150     | 4                | 39                    | 6                 |
| `metaworld/pick-out-of-hole-v2`          | A task where the robot must pick an object out of a hole.                             | 150     | 4                | 39                    | 6                 |
| `metaworld/reach-v2`                     | A task where the robot must reach an object.                                          | 150     | 4                | 39                    | 6                 |
| `metaworld/push-back-v2`                 | A task involving the robot pushing an object backward.                                | 150     | 4                | 39                    | 6                 |
| `metaworld/push-v2`                      | A task where the robot must push an object.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/pick-place-v2`                | A task involving the robot picking up and placing an object.                          | 150     | 4                | 39                    | 6                 |
| `metaworld/plate-slide-v2`               | A task requiring the robot to slide a plate.                                          | 150     | 4                | 39                    | 6                 |
| `metaworld/plate-slide-side-v2`          | A task involving the robot sliding a plate from the side.                             | 150     | 4                | 39                    | 6                 |
| `metaworld/plate-slide-back-v2`          | A task where the robot must slide a plate backward.                                   | 150     | 4                | 39                    | 6                 |
| `metaworld/plate-slide-back-side-v2`     | A task involving the robot sliding a plate backward from the side.                    | 150     | 4                | 39                    | 6                 |
| `metaworld/peg-unplug-side-v2`           | A task where the robot must unplug a peg from the side.                               | 150     | 4                | 39                    | 6                 |
| `metaworld/soccer-v2`                    | A task where the robot must play soccer.                                              | 150     | 4                | 39                    | 6                 |
| `metaworld/stick-push-v2`                | A task involving the robot pushing a stick.                                           | 150     | 4                | 39                    | 6                 |
| `metaworld/stick-pull-v2`                | A task where the robot must pull a stick.                                             | 150     | 4                | 39                    | 6                 |
| `metaworld/push-wall-v2`                 | A task involving the robot pushing against a wall.                                    | 150     | 4                | 39                    | 6                 |
| `metaworld/reach-wall-v2`                | A task where the robot must reach an object with a wall.                              | 150     | 4                | 39                    | 6                 |
| `metaworld/shelf-place-v2`               | A task involving the robot placing an object on a shelf.                              | 150     | 4                | 39                    | 6                 |
| `metaworld/sweep-into-v2`                | A task where the robot must sweep objects into a container.                           | 150     | 4                | 39                    | 6                 |
| `metaworld/sweep-v2`                     | A task requiring the robot to sweep.                                                  | 150     | 4                | 39                    | 6                 |
| `metaworld/window-open-v2`               | A task where the robot must open a window.                                            | 150     | 4                | 39                    | 6                 |
| `metaworld/window-close-v2`              | A task involving the robot closing a window.                                          | 150     | 4                | 39                    | 6                 |

## MP Environments

All envs also exist in MP-variants. Refer to them using `metaworld_ProMP/<name-v2>` or `metaworld_ProDMP/<name-v2>` (DMP is currently not supported as of now).
