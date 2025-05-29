from cog import BasePredictor, Input, Path as CogPath
from typing import List, Union
import subprocess
import time
import logging
import requests
import copy
import shutil
import json
import hashlib
import uuid
from io import BytesIO
import tarfile
from PIL import Image
from pathlib import Path
from urllib.parse import urlparse
import boto3

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

class Predictor(BasePredictor):
    def setup(self):
        # Launch ComfyUI server
        self.server = subprocess.Popen([
            "python", "ComfyUI/main.py",
            "--listen", "0.0.0.0",
            "--port", "8188"
        ])
        host = "http://127.0.0.1:8188"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(host, timeout=5)
                if r.status_code == 200:
                    logging.info("ComfyUI server ready at %s", host)
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("ComfyUI failed to start within 60s")

    def get_bucket_name(self, filename: str, prefix: str) -> str:
        # Original lookup_map for bucket suffixes
        lookup_map = {
            "a": "a1", "b": "b1", "c": "c1",
            "d": "d1", "e": "e1", "f": "f1",
            "A": "a2", "B": "b2", "C": "c2",
            "D": "d2", "E": "e2", "F": "f2",
        }
        # Determine the character to lookup based on filename
        if "_" not in filename:
            lookup_char = filename[0]
        else:
            subfilename = filename[filename.rfind("_") + 1:]
            lookup_char = subfilename[0]
        # Fetch mapped bucket suffix or fallback
        bucket_suffix = lookup_map.get(lookup_char, lookup_char)
        return prefix + bucket_suffix

    def upload_to_digitalocean_spaces(
        self,
        file_path: str,
        bucket_prefix: str,
        region: str,
        access_key: str,
        secret_key: str,
    ) -> str:
        # Upload file to DigitalOcean Spaces
        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=region,
            endpoint_url=f'https://{region}.digitaloceanspaces.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        fname = Path(file_path).name
        bucket = self.get_bucket_name(fname, bucket_prefix)
        obj = hashlib.md5(fname.encode()).hexdigest() + Path(file_path).suffix
        try:
            client.upload_file(file_path, bucket, obj, ExtraArgs={'ACL': 'public-read'})
            return f'https://{bucket}.{region}.cdn.digitaloceanspaces.com/{obj}'
        except Exception as e:
            logging.error(f"DO upload failed: {e}")
            return ''

    def predict(
        self,
        images: str = Input(
            default='[{"name": "", "inputs": {"prompt": "((Close-up headshot)) ((cjw woman)) with a confident smile, wearing a tailored blazer, in front of a neutral background", "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream", "width": "640", "height": "960", "num_steps": 25}}]',
            description="JSON list of image entries with 'name' and 'inputs'"
        ),
        weights: str = Input(
            default="",
            description="Load LoRA weights (tar.gz URL)"
        ),
        api_json: CogPath = Input(
            default=CogPath("workflow_api/face-match-4-5-api.json"),
            description="CogPath to ComfyUI workflow JSON"
        ),
        image1: CogPath = Input(description="First input image"),
        image2: CogPath = Input(description="Second input image"),
        image3: CogPath = Input(description="Third input image"),
        bypass_reactor: bool = Input(default=False, description="Skip ReActor nodes"),
        bypass_upscale_node: bool = Input(default=False, description="Skip upscaling/TTP nodes"),
        poll_interval: float = Input(default=1.0, description="Seconds between polls"),
        timeout: float = Input(default=300.0, description="Completion timeout (s)"),
        log_level: str = Input(default="INFO", description="Logging level"),
        do_settings: str = Input(default="", description="DigitalOcean Spaces JSON settings"),
    ) -> List[Union[str, dict]]:
        """
        Download input images, optionally load LoRA weights, run ComfyUI, and optionally upload to DO Spaces.
        Returns list of local file paths or dicts with 'name' and 'url'.
        """
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s"
        )
        host = "http://127.0.0.1:8188"

        # Load LoRA weights if provided
        if weights.strip():
            r = requests.get(weights, stream=True, timeout=30)
            r.raise_for_status()
            buf = BytesIO(r.content)
            with tarfile.open(fileobj=buf, mode='r:gz') as tar:
                tar.extractall(path="ComfyUI/models/Lora")

        # Parse inputs
        entries = json.loads(images)
        do_cfg = json.loads(do_settings) if do_settings.strip() else {}
        if not entries:
            raise RuntimeError("No image entries provided")

         # Use the three Input() Paths instead of URLs in JSON
        temp_dir = CogPath("input_tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        input_images = []
        for idx, img in enumerate([image1, image2, image3], start=1):
            ext = CogPath(img).suffix or ".png"
            out = temp_dir / f"img{idx}{ext}"
            shutil.copy(img, out)
            input_images.append(out.name)
        upload_images(input_images, str(temp_dir), host)

        # Load workflow
        wf_base = load_workflow(str(api_json))
        results: List[Union[str, dict]] = []

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

            logging.info(f"=== run {run_id} ===")
            wf = copy.deepcopy(wf_base)
            wf = inject_prompts_and_images(wf, pos, neg, images=input_images)
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

            pid = queue_workflow(wf, host, [out_node])
            imgs_info = await_completion(pid, host, poll_interval, timeout)

            dest = CogPath("ComfyUI/output") / run_id
            download_outputs(imgs_info, host, str(dest))

            # Collect outputs
            for img_path in sorted(dest.glob("*.png")):
                fp = str(img_path)
                if do_cfg:
                    url = self.upload_to_digitalocean_spaces(
                        fp,
                        do_cfg.get("bucket_prefix", ""),
                        do_cfg.get("region", "sfo3"),
                        do_cfg.get("access_key_id", ""),
                        do_cfg.get("secret_access_key", ""),
                    )
                    if url:
                        results.append({"name": run_id, "url": url})
                    else:
                        results.append(CogPath(fp))
                else:
                    results.append(CogPath(fp))

        if not results:
            raise RuntimeError("No outputs generated")

        logging.info(f"Returning {len(results)} results")
        return results