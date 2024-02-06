import numpy as np
import copy

class MimicLandmarks():
    def __init__(self, landmarks):
        self.landmark = []

        for i in range(landmarks.shape[0]):
            self.landmark.append(MimicSingleLandmark(landmarks[i, :]))
    
    def parse_to_client(self) -> str:
        ret = []
        for landmark in self.landmark:
            ret.append([float(landmark.x), float(landmark.y), float(landmark.z)])
        return ret

    def project(self, trans_mat):
        for landmark in self.landmark:
            landmark.project(trans_mat)

    def transfer(self, origin, width, height):
        for landmark in self.landmark:
            landmark.transfer(origin, width, height)

    def rotation(self, yaw, pitch, roll):
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        # Define the rotation matrices for each angle
        Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        Rpitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        Rroll = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        # Calculate the composite rotation matrix
        R = np.dot(Rroll, np.dot(Rpitch, Ryaw))
        for landmark in self.landmark:
            landmark.project(R)
        
    def translation(self, x, y, z):
        for landmark in self.landmark:
            landmark.x += x
            landmark.y += y
            landmark.z += z

    def translationOrign(self):
        Center = np.mean(np.array(self.parse_to_client()), axis=0)
        for landmark in self.landmark:
            landmark.x -= Center[0]
            landmark.y -= Center[1]
            landmark.z -= Center[2]
            
    def copy(self):
        # Create a new MimicLandmarks instance
        new_landmarks = MimicLandmarks(np.empty((0, 3)))
        # Deep copy each MimicSingleLandmark
        new_landmarks.landmark = [landmark.copy() for landmark in self.landmark]
        return new_landmarks
    
class MimicSingleLandmark():
    def __init__(self, landmark):
        self.x = landmark[0]
        self.y = landmark[1]
        self.z = landmark[2]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'x  = {self.x}\n y = {self.y}\n z  = {self.z}\n'

    def project(self, trans_mat):
        self.x, self.y, self.z =  np.array([self.x, self.y, self.z]) @ trans_mat @ np.array([[0,0,1], [1,0,0],[0,1,0]])
        # print(self)

    def transfer(self, origin, width, height):
        self.x, self.y = self.x - origin[0], self.y - origin[1]
        self.x, self.y = self.x / width * 0.2, self.y / height * 0.4
        self.x, self.y = self.x + 0.5, self.y + 0.5

    def copy(self):
        # Create a new MimicSingleLandmark with the same coordinates
        return MimicSingleLandmark(np.array([self.x, self.y, self.z]))