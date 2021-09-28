
"""
here, we'll automate the process of starting from a random state and learning from experience
"""

import os
from tensorflow.keras.models import load_model

from model import build_agz_model, build_rollout_policy


def load_hashmap(path):
    if not os.path.exists(path):
        print("hashmap does not exist. creating hashmap.")
        open(path, "w")

    return {
        key: [int(total_games), int(won_games)] for key, total_games, won_games in
        [x.split() for x in open(path, 'r').read().split('\n') if len(x.split()) == 3]
    }


def get_latest_model():

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if not os.path.exists('./models/current_tree_policy.h5'):
        print("tree policy does not exist. creating tree policy.")
        tree_policy = build_agz_model(10, 128, (9, 9, 11))
        tree_policy.save('./models/current_tree_policy.h5', save_format="h5")

    tree_policy = load_model("./models/current_tree_policy.h5")

    if not os.path.exists('./models/current_rollout_policy.h5'):
        print("rollout policy does not exist. creating rollout policy.")
        rollout_policy = build_rollout_policy(4, 32, (9, 9, 3))
        rollout_policy.save('./models/current_rollout_policy.h5', save_format="h5")

    rollout_policy = load_model("./models/current_rollout_policy.h5")

    zobrist_hashmap = load_hashmap("./models/zobrist_hashmap")

    return tree_policy, rollout_policy, zobrist_hashmap


def commence_self_play_training():

    """
    i'm just going to start writing code and then i'll fill in stuff and make adjustments as i go.
    at the top level, i'd like it to be simple and clean and abstracted.
    """

    tree_policy, rollout_policy, zobrist_hashmap = get_latest_model()

    return

    while True:
        mcts = MCTS(game=Go(size=9, komi=6.5), tree_policy=tree_policy, rollout_policy=rollout_policy, hash_map=the_zobrist)
        mcts.play_to_completion(simulations_per_move=81)
        moves = mcts.get_moves()

        dataset = Datasets()
        dataset.assimilate_game(moves, include_symmetries=True)

        tree_batch = dataset.get_tree_batch()
        rollout_batch = dataset.get_rollout_batch()

        tree_policy.fit(tree_batch.features, tree_batch.targets, verbose=1, batch_size=128, epochs=1)
        rollout_policy.fit(rollout_batch.features, rollout_batch.targets, verbose=1, batch_size=128, epochs=1)


if __name__ == "__main__":
    commence_self_play_training()













