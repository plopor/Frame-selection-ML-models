import cv2
from transformers import pipeline, ViTImageProcessor, VisionEncoderDecoderModel, CLIPProcessor, CLIPModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

# for blip captioning and general OCR
# https://huggingface.co/tasks/image-to-text
# https://www.analyticsvidhya.com/blog/2021/12/step-by-step-guide-to-build-image-caption-generator-using-deep-learning/

# for gpt2 captioning
# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

# for frame extraction
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

# for CLIP
# https://huggingface.co/docs/transformers/en/model_doc/clip

# for captions-prompt similarities
# https://huggingface.co/tasks/sentence-similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_frames(video, freq = 30):
    frames = []
    imgs = []

    vid = cv2.VideoCapture(video)
    succ, image = vid.read()
    count = 0

    while succ:
        if count % freq == 0:
            # this does seem to affect the captioning a lot. Most models seem to use RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frames.append(rgb)
            imgs.append(Image.fromarray(rgb))
        succ, image = vid.read()
        count += 1
    vid.release()

    return frames, imgs

def to_text_blip(imgs):
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)
    captions = captioner(imgs)
    return [cap[0]['generated_text'] for cap in captions]

def to_text_gpt2(frames):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # also tried Fast GPT2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    pixel_values = processor(images=frames, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return captions

def get_closest_frame(frames, prompt):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    closest_frame = None
    score = -float("inf")
    
    for frame in frames:
        combined_values = processor(text=[prompt], images=frame, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**combined_values)
                
        curr_score = outputs.logits_per_image
        
        if curr_score > score:
            score = curr_score
            closest_frame = frame
    
    return closest_frame, score

def cosine_sim(captions, prompt):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    print(captions)

    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    captions_emb = model.encode(captions, convert_to_tensor=True)

    similarities = util.cos_sim(prompt_emb, captions_emb)

    return similarities.argmax()

app = Flask(__name__)
CORS(app)

VIDEO_FOLDER = 'videos'
FRAME_FOLDER = 'frames'

@app.route('/frames/<filename>', methods=['GET'])
def get_frame(filename):
    response = send_file(os.path.join(FRAME_FOLDER, filename))
    return response

@app.route('/search', methods=['POST'])
def search_frame():
    video = request.files.get('video')
    prompt = request.form.get('prompt')
    method = request.form.get('method')

    if video and prompt and method:
        file_name = prompt + '_' + method
        video_path = os.path.join(VIDEO_FOLDER, file_name)
        video.save(video_path)
    else:
        return jsonify('Invalid request'), 400

    frames, imgs = get_frames(video_path)

    # standard OCR captioning + sentence embedding sim
    if method == 'blip':
        captions = to_text_blip(imgs)
        # print(captions)
        ind = cosine_sim(captions, prompt)
        result = frames[ind]

    elif method == 'gpt2':
        captions = to_text_gpt2(frames)
        # print(captions)
        ind = cosine_sim(captions, prompt)
        result = frames[ind]

    # CLIP contrastive pretraining
    else:
        result, score = get_closest_frame(frames, prompt)

    final_img = Image.fromarray(result)
    # final_img.show()

    frame_name = prompt + '_' + method + '.png'
    frame_path = os.path.join(FRAME_FOLDER, frame_name)
    final_img.save(frame_path)

    url = f"http://127.0.0.1:5000/frames/{frame_name}"

    return jsonify({'imageUrl': url}), 200

if __name__ == "__main__":
    app.run()
