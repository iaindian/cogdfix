#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path
from urllib.parse import urlencode
import json
import shutil
import requests
import uuid

# Default fallback node ID
DEFAULT_FALLBACK_NODE = "40"


def prepare_output_dir(out_dir: Path) -> None:
    """
    Clears and recreates the output directory.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
        logging.info(f"Cleared output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {out_dir}")


def upload_image(image_path: Path, host: str, upload_name: str) -> None:
    """
    Uploads the input image to ComfyUI under a unique name to avoid caching issues.
    """
    url = f"{host.rstrip('/')}/upload/image"
    with image_path.open('rb') as f:
        files = {'image': (upload_name, f, 'application/octet-stream')}
        data = {'type': 'input', 'overwrite': 'true'}
        resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    logging.info(f"Uploaded image: {upload_name}")


def load_workflow(path: Path) -> dict:
    """
    Loads workflow JSON into a dict of node_id -> node spec.
    """
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def inject_input_image(workflow: dict, filename: str) -> None:
    """
    Sets 'image' input on all LoadImage nodes.
    """
    for node in workflow.values():
        if node.get('class_type') == 'LoadImage':
            inputs = node.setdefault('inputs', {})
            inputs['image'] = filename
            logging.debug(f"Injected image '{filename}' into node {node.get('id')}")


def inject_parameters(workflow: dict, seed=None, steps=None, scale=None, sampler=None) -> None:
    """
    Injects seed and sampler params into RandomNoise and KSampler nodes.
    """
    for node in workflow.values():
        ctype = node.get('class_type', '')
        inputs = node.setdefault('inputs', {})
        if seed is not None and 'RandomNoise' in ctype:
            inputs['noise_seed'] = seed
        if 'KSampler' in ctype:
            if steps is not None:
                inputs['num_steps'] = steps
            if scale is not None:
                inputs['cfg_scale'] = scale
            if sampler is not None:
                inputs['sampler_name'] = sampler


def find_output_node(workflow: dict) -> str:
    """
    Finds node ID for SaveImage or first IMAGE output.
    """
    for node_id, node in workflow.items():
        if node.get('class_type') == 'SaveImage':
            return node_id
    for node_id, node in workflow.items():
        for out in node.get('outputs', {}).values():
            if out.get('type') == 'IMAGE':
                return node_id
    raise RuntimeError("No IMAGE-emitting node found.")


def queue_workflow(workflow: dict, host: str, output_node: str) -> str:
    """
    Queues the workflow and returns prompt_id.
    """
    url = f"{host.rstrip('/')}/prompt"
    resp = requests.post(url, json={'prompt': workflow, 'outputs': [output_node]})
    resp.raise_for_status()
    data = resp.json()
    pid = data.get('prompt_id') or data.get('id')
    if not pid:
        raise RuntimeError(f"No prompt_id returned: {resp.text}")
    logging.info(f"Queued workflow, prompt_id={pid}")
    return str(pid)


def await_completion(prompt_id: str, host: str, interval: float, timeout: float) -> list:
    """
    Polls until images are ready or timeout.
    Prefers 'output' type images.
    """
    url = f"{host.rstrip('/')}/history/{prompt_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(url)
        resp.raise_for_status()
        outs = resp.json().get(prompt_id, {}).get('outputs', {})
        images = [img for node in outs.values() for img in node.get('images', [])]
        if images:
            output_imgs = [i for i in images if i.get('type') == 'output']
            return output_imgs or images
        time.sleep(interval)
    raise TimeoutError(f"Timeout after {timeout}s for prompt {prompt_id}")


def download_output(image: dict, host: str, out_dir: Path) -> None:
    """
    Downloads image, retrying without type on 404.
    """
    params = {'filename': image['filename']}
    if image.get('type'):
        params['type'] = image['type']
    if image.get('subfolder'):
        params['subfolder'] = image['subfolder']
    url = f"{host.rstrip('/')}/view?{urlencode(params)}"
    resp = requests.get(url)
    if resp.status_code == 404 and 'type' in params:
        params.pop('type')
        url = f"{host.rstrip('/')}/view?{urlencode(params)}"
        resp = requests.get(url)
    resp.raise_for_status()
    out_file = out_dir / image['filename']
    out_file.write_bytes(resp.content)
    logging.info(f"Saved image: {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="ComfyUI image-to-image runner")
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-w', '--workflow', default='workflow_api/default-workflow-api.json')
    p.add_argument('--host', default='http://127.0.0.1:8188')
    p.add_argument('-o', '--output-dir', default='comfyui/output')
    p.add_argument('--output-node')
    p.add_argument('--seed', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--scale', type=float)
    p.add_argument('--sampler')
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--timeout', type=float, default=300.0)
    p.add_argument('--log', default='INFO')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format='%(asctime)s %(levelname)s %(message)s')
    inp = Path(args.input)
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")
    out_dir = Path(args.output_dir)
    prepare_output_dir(out_dir)

    # Use a unique upload name to avoid server caching
    unique_name = f"{uuid.uuid4().hex}_{inp.name}"
    upload_image(inp, args.host, unique_name)

    wf = load_workflow(Path(args.workflow))
    inject_input_image(wf, unique_name)
    inject_parameters(wf, seed=args.seed, steps=args.steps, scale=args.scale, sampler=args.sampler)

    out_node = args.output_node or ''
    if not out_node:
        try:
            out_node = find_output_node(wf)
        except RuntimeError:
            logging.warning(f"Falling back to {DEFAULT_FALLBACK_NODE}")
            out_node = DEFAULT_FALLBACK_NODE
    pid = queue_workflow(wf, args.host, out_node)
    images = await_completion(pid, args.host, args.interval, args.timeout)
    download_output(images[0], args.host, out_dir)
    logging.info("Run complete.")

if __name__ == '__main__':
    main()
