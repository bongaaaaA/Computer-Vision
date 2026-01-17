from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
from Utils import get_center_of_bbox, get_bbox_width,get_foot_position

class Tracker:
    """
    Tracker
    =======
    This class handles object detection, tracking, and visualization
    for a football match using YOLO and ByteTrack.

    The tracker is responsible for:
    -------------------------------
    1. Detecting players, referees, and the ball using a YOLO model.
    2. Tracking detected objects across frames using ByteTrack.
    3. Maintaining structured tracking data per frame.
    4. Interpolating missing ball positions.
    5. Computing object positions (center or foot position).
    6. Drawing visual annotations on video frames.
    7. Visualizing team ball possession statistics.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(sekf,tracks):
        """
        Add position information to each tracked object.

        Parameters:
        ----------
        tracks : dict
            Tracking data containing bounding boxes per frame.

        Notes:
        ------
        - Ball position is computed using bounding box center.
        - Player and referee position is computed using foot position.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    #function ball_positions
    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions across frames.

        Parameters:
        ----------
        ball_positions : list
            List of ball detections per frame.

        Returns:
        -------
        list
            Ball detections with interpolated bounding boxes.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        """
        Run YOLO detection on video frames in batches.

        Parameters:
        ----------
        frames : list
            List of video frames.

        Returns:
        -------
        list
            YOLO detections per frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch =self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections +=detections_batch
        return detections
        
    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect and track objects across frames.

        Parameters:
        ----------
        frames : list
            Video frames.
        read_from_stub : bool
            Load tracking data from a saved file if True.
        stub_path : str
            Path to pickle file containing precomputed tracks.

        Returns:
        -------
        dict
            Tracking data structured by object type and frame.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            cls_name_inv = {v:k for k, v in cls_names.items()}

            # convert to detection supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert to goalKeeper to player objects
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper" :
                    detection_supervision.class_id[object_ind] = cls_name_inv["player"]


            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_name_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_name_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_name_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        # if stub_path is not None:
#             with open(stub_path,'wb') as f:
#                 pickle.dump(tracks,f)
        return tracks
            
    def draw_ellipse(self,frame,bbox,color,track_id=None): 
        """
        Draw an ellipse under a detected object.

        Parameters:
        ----------
        frame : np.ndarray
            Video frame.
        bbox : list
            Bounding box [x1, y1, x2, y2].
        color : tuple
            BGR color.
        track_id : int, optional
            Tracking ID to display.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )



        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    

    def draw_traingle(self,frame,bbox,color):
        """
        Draw a triangle marker above an object.

        Used mainly to highlight the ball or ball possession.
        """
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame


    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self,video_frames, tracks, team_ball_control):
        """
        Draw all annotations on video frames.

        Includes:
        ---------
        - Players (ellipse + ID)
        - Referees
        - Ball
        - Ball possession indicator
        - Team ball control statistics
        """
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255)) 
            
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames


