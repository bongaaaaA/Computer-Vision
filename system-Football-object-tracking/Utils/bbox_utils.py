def get_center_of_bbox(bbox):
    """
    Calculate the center point of a bounding box.
    Parameters:
        bbox (list | tuple): Bounding box coordinates in the format
                             [x1, y1, x2, y2].
    Returns:
        tuple: (x_center, y_center) as integer pixel coordinates.
    Usage:
        Used for objects where the geometric center represents
        the object position (e.g., ball).
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2 ), int((y1+y2)/2)

def get_bbox_width(bbox):
    """
    Compute the width of a bounding box.
    Parameters:
        bbox (list | tuple): Bounding box coordinates
                             [x1, y1, x2, y2].
    Returns:
        int | float: Width of the bounding box in pixels.

    Usage:
        Commonly used for drawing scaled annotations
        (e.g., ellipse under player).
    """
    return bbox[2] - bbox[0]

def measure_distance(p1,p2):
    """
    Usage:
        Useful for measuring player-to-ball distance,
        speed estimation, and proximity-based logic.
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def measure_xy_distance(p1,p2):
    """
    Returns:
        tuple: (x_foot, y_foot) as integer pixel coordinates.

    Usage:
        Used for player position tracking since the foot
        location is more accurate than the center point
        for football analytics.
    """

    return p1[0]-p2[0],p1[1]-p2[1]




def get_foot_position(bbox):
    """
    Estimate the foot (ground contact) position of a player
    from a bounding box.

    Usage:
        Used for player position tracking since the foot
        location is more accurate than the center point
        for football analytics.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)