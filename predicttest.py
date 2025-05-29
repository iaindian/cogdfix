from pathlib import Path
import json
import requests
import copy


from comfyrunbatch import (
    load_prompts,
    find_input_images,
    upload_images,
    load_workflow,
    inject_prompts_and_images,
    inject_parameters,
    strip_reactor_nodes,
    bypass_upscale,
    queue_workflow,
    await_completion,
    download_outputs,
)

def download_to_temp(url, filename):
    r = requests.get(url)
    r.raise_for_status()
    path = Path("test_inputs")
    path.mkdir(exist_ok=True)
    out_path = path / filename
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path



# Download the same image three times (or any test image you want)
img_url = "https://replicate.delivery/xezq/pIjKkbQdmNISJJvR6JFVNCfyn6reewmXeJDDsbINZXJxPrVSB/ComfyUI_00001_.webp"
image1 = download_to_temp(img_url, "img1.webp")
image2 = download_to_temp(img_url, "img2.webp")
image3 = download_to_temp(img_url, "img3.webp")

images_json = json.dumps([
    {
        "name": "test-beru1",
        "inputs": {
            "prompt": "((Close-up headshot)) ((cjw woman)) with a confident smile, wearing a tailored blazer, in front of a neutral background",
            "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream",
            "width": "640",
            "height": "960"
        }
    }
])

def predict(images,
    weights,
    api_json,
    image1,
    image2,
    image3,
    bypass_reactor,
    bypass_upscale_node,
    poll_interval,
    timeout,
    log_level,
    do_settings):
    
    wf_base = load_workflow(str(api_json))
    entries = json.loads(images)
    
    for entry in entries:
        run_id = entry.get("name") or str(uuid.uuid4())
        inp = entry.get("inputs", {})
        pos = inp.get("prompt", "")
        neg = inp.get("negative_prompt", "")
        seed = int(inp.get("seed", 767))
        guidance = float(inp.get("guidance_scale", 3.7))
        steps = int(inp.get("num_inference_steps", inp.get("num_steps", 33)))
        width = int(inp.get("width", 896))
        height = int(inp.get("height", 1152))
        strength = float(inp.get("strength", 1.0))
        scheduler = inp.get("scheduler")

        wf = copy.deepcopy(wf_base)
        wf = inject_prompts_and_images(wf, pos, neg, images=["1.png","2.png","3.png"])
        wf = inject_parameters(
            wf,
            seed=seed,
            guidance_scale=guidance,
            num_steps=steps,
            width=width,
            height=height,
            strength=strength,
            scheduler=scheduler,
        )
        if bypass_reactor:
            wf = strip_reactor_nodes(wf)
        if bypass_upscale_node:
            wf = bypass_upscale(wf)
            out_node = "230"
        else:
            out_node = "230"
       
        print(json.dumps(wf,indent=2))
        
results = predict(
    images=images_json,
    weights="",
    api_json="workflow_api/face-match-4-5-api.json",
    image1=image1,
    image2=image2,
    image3=image3,
    bypass_reactor=False,
    bypass_upscale_node=True,
    poll_interval=1.0,
    timeout=300.0,
    log_level="INFO",
    do_settings=""
)

