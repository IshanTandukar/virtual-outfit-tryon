import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File,HTTPException
from fastapi.responses import StreamingResponse
from models import SegGenerator, GMM, ALIASGenerator
from utilss import load_checkpoint
from helper import get_opt,get_image
from model_helper import segmentation_generation,clothes_deformation,try_on_synthesis
from load import load_data
from io import BytesIO
from PIL import Image
from cloth_mask_model import predic
print(1)
from simple_extractor import parsing
print(2)
from nayapose import get_pose_key_and_json
import numpy as np


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

opt = get_opt()

seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
opt.semantic_nc = 7
alias = ALIASGenerator(opt, input_nc=9)
opt.semantic_nc = 13

load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

seg.eval()
gmm.eval()
alias.eval()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Virtual Tryon2"}

# @app.post("/tryon/")
# async def virtual_tryon(
#     human_image: UploadFile = File(...),
#     cloth_image: UploadFile = File(...)
# ):
    
#     try:
#         if not human_image.content_type.startswith('image/') or \
#            not cloth_image.content_type.startswith('image/'):
#             raise HTTPException(400, detail="Invalid file type")

#         human_filename = human_image.filename
        
#         image_path = os.path.join("datasetss/test/image", human_filename)
#         if not os.path.exists(image_path):
#             raise HTTPException(
#                 400,
#                 detail=f"Human image {human_filename} not found in dataset. Please ensure the image exists in datasets/test/image/"
#             )
#         image, image_parse, openpose_img, openpose_json = get_image_paths(human_filename)

#         cloth = Image.open(cloth_image.file)
#         cloth_mask = predic(cloth)

#         new_parse_agnostic_map, pose_rgb, cm, c, img_agnostic = load_data(
#             openpose_img, openpose_json, image_parse, cloth, cloth_mask, image
#         )
        
#         parse, pose_rgb = segmentation_generation(opt, new_parse_agnostic_map, pose_rgb, seg, cm, c)
#         warped_c, warped_cm = clothes_deformation(img_agnostic, parse, pose_rgb, gmm, cm, c)
#         im = try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm)

#         img_bytes = BytesIO()
#         im.save(img_bytes, format="JPEG")
#         img_bytes.seek(0)
        
#         return StreamingResponse(img_bytes, media_type="image/jpeg")
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/tryon/")
async def virtual_tryon(
    human_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    try:

        if not human_image.content_type.startswith('image/') or \
           not cloth_image.content_type.startswith('image/'):
            raise HTTPException(400, detail="Invalid file type")

        human_filename = human_image.filename
        print(human_filename)
        human_image = await human_image.read()
        human_image = Image.open(BytesIO(human_image))
        print("HumankoImage:",human_image)

        cloth_image = await cloth_image.read()
        cloth_image = Image.open(BytesIO(cloth_image))
        print("Clothko image:",cloth_image)
        
        image_path = os.path.join("datasetss/test/image", human_filename)
        if os.path.exists(image_path):
            human_image, image_parse, openpose_img, openpose_json = get_image(human_filename)
            print("Human Image:",human_image)
            print("Image Parse:",image_parse)
            print("Openpose Image:",openpose_img)
            print("Openpose JSON:",openpose_json)
        else:
            image_parse = parsing(human_image)
            openpose_img, openpose_json = get_pose_key_and_json(human_image)
            print(openpose_json)
            pose_data = openpose_json['people'][0]['pose_keypoints_2d']
            print(pose_data)
            # pose_data = np.array(pose_data)
            def omit_every_third_element(input_list):
                return [input_list[i:i+2] for i in range(0, len(input_list), 3)]
            new_list = omit_every_third_element(pose_data)

            openpose_json = new_list
            openpose_json = np.array(openpose_json)
            print(openpose_json)
        # cloth = Image.open(cloth_image.file)
        cloth_mask = predic(cloth_image)

        new_parse_agnostic_map, pose_rgb, cm, c, img_agnostic = load_data(
            openpose_img, openpose_json, image_parse, cloth_image, cloth_mask, human_image
        )
        
        parse, pose_rgb = segmentation_generation(opt, new_parse_agnostic_map, pose_rgb, seg, cm, c)

        warped_c, warped_cm = clothes_deformation(img_agnostic, parse, pose_rgb, gmm, cm, c)
        im = try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm)

        img_bytes = BytesIO()
        im.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/test/")
async def test(human_image: UploadFile = File(...)):
    image = await human_image.read()
    print(image)
    image = Image.open(BytesIO(image))
    print(image)
    image.save("pawan.jpg")
    return {"message": "Success"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)