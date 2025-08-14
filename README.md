# Neural Style Transfer with OpenCV

Transform your images and videos into stunning works of art using neural style transfer powered by OpenCV's Deep Neural Network module.

### requirements

- Python 3.6+
- OpenCV 4.0+
- imutils
- NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-style-transfer-opencv.git
cd neural-style-transfer-opencv
```

2. Install required packages:
```bash
pip install opencv-python imutils numpy argparse
```

3. Download pre-trained style transfer models (`.t7` files) and place them in a `models/` directory.

##  Usage

### Single Image Style Transfer

Apply a specific style to an image:

```bash
python neural_style_transfer.py --image images/your_image.jpg --model models/starry_night.t7
```

### Compare Multiple Models

Test different styles on the same image:

```bash
python neural_style_transfer_examine.py --models models/ --image images/your_image.jpg
```

Press any key to cycle through different style models.

### Video Style Transfer

Transform an entire video file:

```bash
python neural_style_transfer_video_offline.py --model models/the_scream.t7 --video videos/your_video.mp4
```

### Real-time Camera Style Transfer

Start live style transfer with your webcam:

```bash
python neural_style_transfer_video.py --models models/
```

**Controls:**
- Press `n` to switch to the next style model
- Press `q` to quit

##  Project Structure

```
neural-style-transfer-opencv/
├── neural_style_transfer.py              # Single image processing
├── neural_style_transfer_examine.py      # Model comparison tool
├── neural_style_transfer_video_offline.py # Video file processing
├── neural_style_transfer_video.py        # Real-time camera feed
├── models/                               # Style transfer models (.t7 files)
├── images/                               # Input images
├── videos/                               # Input videos
├── output/                               # Generated results
└── README.md
```

##  Technical Details

### Model Format
This toolkit uses Torch (.t7) format neural style transfer models. Popular options include:
- Starry Night
- The Scream
- Mosaic
- Rain Princess
- Udnie

### Image Processing Pipeline
1. Load and preprocess input image/frame
2. Create blob with mean subtraction (103.939, 116.779, 123.680)
3. Forward pass through neural network
4. Post-process output tensor
5. Save or display result




##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Original neural style transfer research by [Gatys et al.](https://arxiv.org/abs/1508.06576)


