sample_rate = 32000
<<<<<<< HEAD
"""int: Target sample rate during feature extraction"""

window_size = 2048
"""int: Size of FFT window"""

overlap = 1024
"""int: Amount of overlap between frames"""

time_steps = 128
"""int: The time steps of a specrogram patch for training"""

test_hop_frames = 16

mel_bins = 64
"""int: Number of Mel bins"""

kmax = 3

labels = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 
          'Burping_or_eructation', 'Bus', 'Cello', 'Chime', 'Clarinet', 
          'Computer_keyboard', 'Cough', 'Cowbell', 'Double_bass', 
          'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping', 
          'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 
          'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 
          'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors', 'Shatter', 
          'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone', 
          'Trumpet', 'Violin_or_fiddle', 'Writing']
=======
"""number: Target sample rate during feature extraction."""

window_size = 2048
"""int: Size of FFT window."""

overlap = 1024
"""int: Amount of overlap between frames."""

mel_bins = 64
"""int: Number of Mel bins."""

time_steps = 32

kmax = 3

labels = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Burping_or_eructation', 'Bus', 'Cello', 'Chime', 'Clarinet', 'Computer_keyboard', 'Cough', 'Cowbell', 'Double_bass', 'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping', 'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors', 'Shatter', 'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone', 'Trumpet', 'Violin_or_fiddle', 'Writing']
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600

lb_to_ix = {lb: i for i, lb in enumerate(labels)}
ix_to_lb = {i: lb for i, lb in enumerate(labels)}

<<<<<<< HEAD
corrupted_files = ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']
=======
num_classes = len(labels)
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
