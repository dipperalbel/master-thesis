import numpy as np

ucf_2_full = {
    'RockClimbingIndoor': 'climb',
    'RopeClimbing': 'climb',
    'Fencing': 'fencing',
    'GolfSwing': 'golf',
    'SoccerPenalty': 'kick_ball',
    'PullUps': 'pullup',
    'Punch': 'punch',
    'BoxingPunchingBag': 'punch',
    'BoxingSpeedBag': 'punch',
    'PushUps': 'pushup',
    'Biking': 'ride_bike',
    'HorseRiding': 'ride_horse',
    'Basketball': 'shoot_ball',
    'Archery': 'shoot_bow',
    'WalkingWithDog': 'walk'
}

hmdb_2_full = {
    'climb': 'climb',
    'fencing': 'fencing',
    'golf': 'golf',
    'kick_ball': 'kick_ball',
    'pullup': 'pullup',
    'punch': 'punch',
    'pushup': 'pushup',
    'ride_bike': 'ride_bike',
    'ride_horse': 'ride_horse',
    'shoot_ball': 'shoot_ball',
    'shoot_bow': 'shoot_bow',
    'walk': 'walk'
}

ucf_hmdb_full_classes = [
    'climb',
    'fencing',
    'golf',
    'kick_ball',
    'pullup',
    'punch',
    'pushup',
    'ride_bike',
    'ride_horse',
    'shoot_ball',
    'shoot_bow',
    'walk'
    ]

ucf_hmdb_full_ohe = {x: np.zeros(len(ucf_hmdb_full_classes)) for x in ucf_hmdb_full_classes}
for i, key in enumerate(ucf_hmdb_full_ohe):
    ucf_hmdb_full_ohe[key][i] = 1