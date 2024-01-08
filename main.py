import time
from fastsam import FastSAM
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
import av
import io
import math

model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cuda'
VIDEO_PATH = ""

video = cv2.VideoCapture(VIDEO_PATH)

if video.isOpened():
    input = av.open(VIDEO_PATH)
    frame_count = math.floor(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(video.get(cv2.CAP_PROP_FPS), 2)
    width = math.floor(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = math.floor(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_memory_file = io.BytesIO()
    output = av.open(output_memory_file, 'w', format="mp4")
    output.add_stream(template=input.streams.audio[0])
    stream = output.add_stream('h264', str(fps))
    stream.width = width  # Set frame width
    stream.height = height  # Set frame height
    stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
    stream.options = {'crf': '17'}


    def get_rotoscope(annotations):
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        
        rotoscope = Image.new('RGB', (width, height), (255, 255, 255))

        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()

        contour_all = []
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            annotation = cv2.resize(
                annotation,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
                )
            contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)

        for contour in contour_all:
            prev_x = contour[0][0][0]
            prev_y = contour[0][0][1]
            for points in contour:
                for point in points:
                    x = point[0]
                    y = point[1]

                    draw = ImageDraw.Draw(rotoscope)
                    draw.line((prev_x, prev_y, x, y), fill="BLACK", width=3)

                    prev_x = x
                    prev_y = y
        
        return rotoscope

    for i in range(frame_count):
        start = time.time()

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        
        im_pil = Image.fromarray(frame)

        everything_results = model(im_pil, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        
        if everything_results[0] == None:
            ann = []
        else: 
            ann = everything_results[0].masks.data
        
        start_roto = time.time()
        rotoscope = get_rotoscope(ann)

        final_frame = av.VideoFrame.from_image(rotoscope)
        packet = stream.encode(final_frame)  # Encode video frame
        output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).

        print("Frame %s / %s done in %s seconds" % (i + 1, frame_count, time.time() - start))

    # Flush the encoder
    packet = stream.encode(None)
    output.mux(packet)

    output.close()

    # Write BytesIO from RAM to file, for testing
    with open("output/out.mp4", "wb") as f:
        f.write(output_memory_file.getbuffer())
else:
    print("Video not found")
