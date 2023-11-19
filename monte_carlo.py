import numpy as np

CARD_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}

def calculate_hand_value(hand):
    """Calculate the value of a blackjack hand."""
    value = sum(CARD_VALUES[card] for card in hand)

    num_aces = hand.count('A')
    while value > 21 and num_aces:
        value -= 10
        num_aces -= 1
    return value

def play_blackjack(policy, num_episodes=100000):
    """Simulate blackjack games using the Monte Carlo method."""
    state_action_values = {}
    state_visit_counts = {}

    for _ in range(num_episodes):
        player_hand = ['A', '2']
        dealer_hand = ['K']
        while calculate_hand_value(player_hand) < 12:
            player_hand.append(np.random.choice(list(CARD_VALUES.keys())))
        
        while calculate_hand_value(dealer_hand) < 17:
            dealer_hand.append(np.random.choice(list(CARD_VALUES.keys())))

        player_value = calculate_hand_value(player_hand)
        dealer_value = calculate_hand_value(dealer_hand)

        if player_value > 21 or (dealer_value <= 21 and dealer_value >= player_value):
            reward = -1  
        elif player_value == dealer_value:
            reward = 0 
        else:
            reward = 1   
        state = (calculate_hand_value(player_hand), dealer_hand[0], 'usable' if 'A' in player_hand else 'not-usable')
        action = policy(state)
        state_action = (state, action)

        state_visit_counts[state_action] = state_visit_counts.get(state_action, 0) + 1
        state_action_values[state_action] = state_action_values.get(state_action, 0) + (reward - state_action_values.get(state_action, 0)) / state_visit_counts[state_action]

    return state_action_values

def simple_policy(state):
    """A simple policy for the player."""
    player_value, dealer_card, usable_ace = state
    return 'hit' if player_value < 20 else 'stick'

if __name__ == "__main__":
    policy = simple_policy
    state_action_values = play_blackjack(policy)

    for key, value in state_action_values.items():
        print(f"{key}: {value}")
