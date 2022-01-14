import glob
import torch
import os
import cv2 as cv
import numpy as np
from utils.center_crop import center_crop

class action_dataset():
    def __init__(self, T, end_paths, T_eval=None, small=False):
        """

        This is the dataset for the RGB scenes. These scenes are loaded directly as a video from disk.

        The init generates the list of available sequences to be used by the dataloader.

        Args:
            T: The length of each sequence of frames, normally between 32 - 64

        """
        # The name of the classes
        self.classes = sorted(os.listdir('dataset'))
        # Generate a vector for each class, just like one hot encoding.
        self.int_classes = np.eye(len(self.classes))
        self.T = T
        self.T_eval = T_eval if T_eval is not None else T
        self.T_min = 32
        self.classes_size = {x: 0 for x in self.classes}
        self.total = 0
        self.evaluate = True if end_paths == ['/test'] or len(end_paths)==2 else False

        self.end_path = end_paths

        # If training with small network
        self.small = small

        # Random pad, lengthens the video by the random pad value, so theres some leeway for choosing different frames
        # as start of the sequence
        self.random_pad_value = 0.15
        self.random_pad = int(self.T * self.random_pad_value)

        # Chances of transforms 0 to 1
        self.flip_chance = 0.5
        self.noise_chance = 0.5
        self.add_chance = 0.5
        self.multiply_chance = 0.5

        if self.evaluate:
            self.flip_chance = 0
            self.noise_chance = 0
            self.add_chance = 0
            self.multiply_chance = 0

        # Gaussian noise added
        self.gaussian_noise = 0.1
        # Max per pixel addition
        self.add_max = 0.18
        # Max per pixel multiplication
        self.multiply_max = 0.4

        # Reads all the videos
        self.video_list = []
        for path in self.end_path:
            for i_class, int_class in zip(self.classes, self.int_classes):
                self.videos = glob.glob('dataset/' + i_class + path + '/*.avi')
                for video in self.videos:

                    cap = cv.VideoCapture(video)
                    frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    
                    # If the video is shorter than the minimum size, use it with its own size
                    if frames > self.T_min:
                        self.classes_size[i_class] += 1
                        self.total += 1
                        start = 0
                        end = frames
                        self.t_video_dict = {'class name': i_class, 'class int': int_class, 'video path': video,
                                            'frames': frames, 'start': start, 'end': end}

                        self.video_list.append(self.t_video_dict)
        print('Evaluation Dataset' if self.evaluate else 'Training Dataset')
        print('Classes: ', self.classes_size)
        self.pos_weight = torch.zeros(len(self.classes))
        for i, key in enumerate(self.classes_size):
            self.pos_weight[i] = self.total/self.classes_size[key]
        print('Positions Weights: ', self.pos_weight)
        
    def get_start_end(self, frames):
        if self.evaluate:
            T = self.T_eval
        else:
            T = self.T
        
        slack = int(np.clip(frames - T, 0, np.inf))+5

        start = np.random.randint(0, slack+1)
        end = np.clip(start + T, -np.inf, frames)

        return start, end

    def get_class_int(self, name):
        class_int = None
        for index, class_name in enumerate(self.classes):
            if name == class_name:
                class_int = self.int_classes[index]
                break
        return class_int

    def pad_remaining_frames(self, image):
        if self.evaluate:
            T = self.T_eval
        else:
            T = self.T

        t, h, w, c = image.shape

        frame = np.zeros([T, h, w, c])

        start = T - t
        end = start + T
        frame[start:end, :, :] = image
        return frame



    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Args:
            idx: returns the scene based on this index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        size = 226.
        # Start the capture of the video
        cap = cv.VideoCapture(self.video_list[idx]['video path'])

        start, end = self.get_start_end(frames = self.video_list[idx]['frames'])

        if not self.evaluate:
            crop_offset = np.random.randint(-8, 8, size=2)
        else:
            crop_offset = None
        list_frame = []

        # while there are frames available
        i = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret and i >= start:
                w, h, c = frame.shape
                if w < size or h < size:
                    d = size - min(w, h)
                    sc = 1 + d / min(w, h)
                    frame = cv.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
                
                # If on small training, resize image
                if self.small:
                    frame = cv.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
                    crop_size = 110
                else:
                    crop_size = 224
                frame = (frame / 255.) * 2 - 1
                frame = center_crop(frame, crop_size, offset=crop_offset)
                list_frame.append(frame)
            i += 1
            if i >= end:
                break

        # Turn video into array
        image = np.asarray(list_frame)

        class_name = self.video_list[idx]['class name']
        class_int = self.video_list[idx]['class int']
        image_transformed = image.copy()

        # Transform the video randomly
        if not self.evaluate:
            image_transformed = self.transform(image).copy()
            if self.video_list[idx]['class name'] == 'sit':
                reverse_chance = np.random.random()
                if reverse_chance > 0.5:
                    image_transformed = np.flip(image_transformed, axis=0)
                    class_name = 'stand'
                    class_int = self.get_class_int(class_name)
                    image = np.flip(image, axis=0)
            elif self.video_list[idx]['class name'] == 'wave':
                reverse_chance = np.random.random()
                if reverse_chance > 0.5:
                    image_transformed = np.flip(image_transformed, axis=0)
                    class_name = 'wave'
                    class_int = self.get_class_int(class_name)
                    image = np.flip(image, axis=0)
            elif self.video_list[idx]['class name'] == 'stand':
                reverse_chance = np.random.random()
                if reverse_chance > 0.5:
                    image_transformed = np.flip(image_transformed, axis=0)
                    class_name = 'sit'
                    class_int = self.get_class_int(class_name)
                    image = np.flip(image, axis=0)

        image = self.pad_remaining_frames(image)
        image_transformed = self.pad_remaining_frames(image_transformed)
        # Save into dictionary
        sample = {'video images': image_transformed.copy(),
                  'class': class_name,
                  'integer class': class_int,
                  'real frame': image.copy()}

        return sample

    def transform(self, image):
        """
        Transforms the image, adding noise and flipping the image in the left-right direction, given a random chance

        Args:
            image: image to be transformed
        """
        noise = np.random.random()
        if noise < self.noise_chance:
            image = self.add_noise(image)
        flip = np.random.random()
        if flip < self.flip_chance:
            image = self.flip_lr(image)
        multiply = np.random.random()
        if multiply < self.multiply_chance:
            image = self.multiply(image)
        add = np.random.random()
        if add < self.add_chance:
            image = self.add(image)
        return image

    def add(self, image):
        """
        Adds a single value to the whole image
        :param image: image to be added
        :return: transformed image
        """
        image = image + (np.random.random()-0.5)*2*self.add_max
        image = np.clip(image, -1, 1)
        return image

    def multiply(self, image):
        """
        multiply the image with a single value
        :param image: image to be multiplied
        :return: transformed image
        """
        image = image * (1+(np.random.random()-0.5)*2*self.multiply_max)
        image = np.clip(image, -1, 1)
        return image

    def add_noise(self, image):
        """
        Adds gaussian noise to the image, given standard deviation
        Args:
            image: image to be noise added
        """
        image = np.random.normal(image, self.gaussian_noise)
        image = np.clip(image, -1, 1)
        return image

    def flip_lr(self, image):
        """
        Flips the image as a mirror
        Args:
            image: image to be flipped in the vertical axis
        """
        image = np.flip(image, axis=2)
        return image



if __name__ == '__main__':
    a = action_dataset(T = 64, end_paths=['/test', '/training'])
    print(len(a))
    while True:
        in_dict = a[np.random.randint(0, len(a))]
        for image in in_dict['video images']:
            cv.imshow('teste', (image+1)/2)

            cv.waitKey(int(1/12*1000))
