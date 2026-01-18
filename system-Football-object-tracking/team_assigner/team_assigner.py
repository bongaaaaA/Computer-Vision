from sklearn.cluster import KMeans

class TeamAssigner:
    """
    TeamAssigner
    ============
    This class assigns each detected player to a team based on jersey color
    using unsupervised clustering (K-Means).

    The approach works in two stages:
    1. Extract dominant jersey color for each player from the upper half
       of the bounding box.
    2. Cluster all player colors into two teams and assign team IDs.

    Main Responsibilities:
    ----------------------
    1. Extract player jersey color using pixel clustering.
    2. Learn team color centroids automatically.
    3. Assign a consistent team ID to each player across frames.
    """
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        """
        Create and fit a K-Means clustering model on image pixels.

        Parameters:
        ----------
        image : np.ndarray
            Image patch containing player jersey (H x W x 3).

        Returns:
        -------
        KMeans
            Trained K-Means model with 2 clusters.
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self,frame,bbox):
        """
        Extract the dominant jersey color of a player.

        Parameters:
        ----------
        frame : np.ndarray
            Original video frame.
        bbox : list or tuple
            Player bounding box [x1, y1, x2, y2].

        Returns:
        -------
        np.ndarray
            RGB color vector representing the player's jersey color.
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    def assign_team_color(self,frame, player_detections):
        """
        Learn team colors by clustering all player jersey colors.

        Parameters:
        ----------
        frame : np.ndarray
            Current video frame.
        player_detections : dict
            Dictionary of detected players with bounding boxes.

        Notes:
        ------
        This method should be called once (or few times) to initialize
        team color centroids.
        """
        
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        """
        Assign a team ID to a player.

        Parameters:
        ----------
        frame : np.ndarray
            Current video frame.
        player_bbox : list or tuple
            Player bounding box [x1, y1, x2, y2].
        player_id : int
            Unique player tracking ID.

        Returns:
        -------
        int
            Team ID (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id