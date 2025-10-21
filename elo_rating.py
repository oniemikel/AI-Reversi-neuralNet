# elo_rating.py

import json


class EloRating:
    def __init__(self, initial_rating=1200, k_factor=30):
        self.ratings = {}
        self.k = k_factor

    def add_player(self, name, rating=None):
        self.ratings[name] = rating if rating is not None else 1200

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a, player_b, score_a):
        rating_a = self.ratings.get(player_a, 1200)
        rating_b = self.ratings.get(player_b, 1200)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        new_rating_a = rating_a + self.k * (score_a - expected_a)
        new_rating_b = rating_b + self.k * ((1 - score_a) - expected_b)

        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b

    def get_rating(self, player):
        return self.ratings.get(player, 1200)

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.ratings, f, indent=2)

    def load(self, filepath):
        with open(filepath, "r") as f:
            self.ratings = json.load(f)


def update_elo_from_match_results(elo, player_a, player_b, results):
    """
    results: list of outcomes from player_a's視点 (1=win, 0.5=draw, 0=loss)
    """
    score_sum = sum(results)
    n = len(results)
    score_avg = score_sum / n
    elo.update_ratings(player_a, player_b, score_avg)
