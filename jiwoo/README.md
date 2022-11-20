# Hungry Geese
## dqn 코드로 작성된 노트북을 double dqn 모델로 수정 후 train 시킴

dqn의 경우 q-value가 과대평가 되는 문제가 있다. 
이는 dqn이 q function에서 max()를 이용하기 때문이다.

이를 해결하기 위해 Double DQN은 2개의 다른 네트워크(DQN과 Target Network)를 사용한다. 

Deep Q Network ; 다음 상태의 최대 Q- 값으로 최상의 행동의 선택
Target Network ; 1. 선택된 행동 한에서 추정된 q-value를 계산
                 2. Target로부터 추정된 q-value를 기반으로 DQN을 업데이트
                 3. 규칙적으로 DQN의 파라미터를 기반으로 Target Network 파라미터를 업데이트


아래의 코드는 pytorch를 바탕으로 작성된 것이지만, 코드 흐름을 참고아여 kaggle notebook을 수정하였다. 

import torch
from torch import nn


def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions


def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)        
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values


def q_learning_update(states: torch.Tensor,
                      rewards: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float,
                      q_network: nn.Module) -> torch.Tensor:
    """Q-Learning update with explicitly decoupled action selection and evaluation steps."""
    actions = select_greedy_actions(states, q_network)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network)
    return q_values

def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    """Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q_network_1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network_2)
    return q_values
