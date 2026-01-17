from Utils import read_video , save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    """
    Main pipeline for football video analysis and tracking.

    This function orchestrates the complete workflow:
    1. Read input video frames.
    2. Detect and track players, referees, and the ball using YOLO + ByteTrack.
    3. Estimate and compensate for camera movement.
    4. Transform player positions to a top-down (bird's-eye) view.
    5. Interpolate missing ball detections.
    6. Estimate player speed and total distance covered.
    7. Assign players to teams based on jersey color clustering.
    8. Assign ball possession to the nearest player.
    9. Visualize tracking results, team control, speed, and distance.
    10. Save the final annotated video to disk.

    The pipeline is designed to be modular, allowing each component
    (tracking, camera motion, team assignment, etc.) to be developed
    and tested independently.
    """
    # -----------------------------
    # Read input video
    # -----------------------------
    video_frames = read_video('input_vedio/test.mp4')
    # -----------------------------
    # Initialize tracker and load tracks
    # -----------------------------
    tracker = Tracker('Models/best.pt')

    tracks = tracker.get_object_track(video_frames,
                                       read_from_stub=True , stub_path='stubs/track_stubs.pkl')
    #get oject position
    tracker.add_position_to_tracks(tracks)

    # -----------------------------
    # Camera movement estimation
    # -----------------------------r
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # -----------------------------
    # View (perspective) transformation
    # -----------------------------
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # -----------------------------
    # Interpolate missing ball positions
    # -----------------------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # -----------------------------
    # Speed and distance estimation
    # -----------------------------
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # -----------------------------
    # Team assignment (color-based clustering)
    # -----------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # -----------------------------
    # Ball possession assignment
    # -----------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player  = player_assigner.assign_ball_to_player(player_track,
                                                                 ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)



    # -----------------------------
    # Draw annotations
    # -----------------------------
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)    

    # -----------------------------
    # Save output video
    # -----------------------------
    save_video(output_video_frames, 'output_video/output_video.avi')


if __name__ == '__main__':
    main()


