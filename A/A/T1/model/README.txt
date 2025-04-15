README - Hex AI

1. 项目介绍

本项目实现了一个基于 Minimax 和 MCTS（蒙特卡洛树搜索）算法的 Hex 棋 AI。

3. 环境要求

该项目需要 Python 3.8 或更高版本，并依赖以下库：

numpy

multiprocessing（Python 标准库，默认包含）

4. 安装依赖

在 Windows 平台上，使用以下命令安装所需依赖：

pip install numpy

5. 运行模型

在 附件/ 目录下，使用以下命令运行 Hex AI：

python -m model.hex_ai

6. 代码概述

HexAI 类：核心 AI 逻辑，包括 Minimax 和 MCTS 算法。

simulate_random_game_static：随机模拟游戏，评估局势。

evaluate：综合评估局面。

get_best_move：获取当前最优落子点。

7. 其他说明

AI 可通过 Minimax 或 MCTS 进行决策。

代码已针对 Windows 平台优化，支持多进程计算。

注明：

Hex AI 使用 multiprocessing.cpu_count() 自动检测可用 CPU 核心，并根据核心数量调整搜索算法的计算规模。如果你的 CPU 线程较少，AI 运行时间可能会显著增加。

Hex AI 采用多进程并行计算，默认使用 multiprocessing.cpu_count() 获取所有可用核心

核心数越多，计算速度越快，但在低性能 CPU 上可能运行缓慢（可能运行超过10s）。但是本机在测试时没有出现超时问题（16核）
