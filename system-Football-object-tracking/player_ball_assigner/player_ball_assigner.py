import sys 
sys.path.append('../')
from Utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    """
    PlayerBallAssigner
    ==================
    This class assigns the ball to the closest player based on spatial distance.

    The assignment is performed by measuring the distance between the ball center
    and the bottom corners of each player's bounding box, which approximately
    represent the player's feet position on the ground.

    Main Responsibilities:
    ----------------------
    1. Compute the center point of the detected ball bounding box.
    2. Measure distances between the ball and each player's feet.
    3. Assign ball possession to the closest player within a distance threshold.
    """
    def __init__(self):
        self.max_player_ball_distance = 70
    def assign_ball_to_player(self,players,ball_bbox):
        """
        Assign the ball to the most likely player.

        Parameters:
        ----------
        players : dict
            Dictionary of detected players in the current frame.
            Expected format:
            {
                player_id: {
                    'bbox': [x1, y1, x2, y2]
                },
                ...
            }

        ball_bbox : list or tuple
            Bounding box of the detected ball [x1, y1, x2, y2].

        Returns:
        -------
        int
            The player_id of the assigned player.
            Returns -1 if no suitable player is found.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player