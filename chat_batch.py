import os, csv, re, json, torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# —— 和你原来一致 —— #
PROMPT = """### 背景 ###
请识别图像中的地物类型，包括建筑物、道路、植被等，识别其类型、位置、大小等特征。
### 输出格式 ###
您的输出由以下两部分组成，确保包含这两部分:
### 思考 ###
观察图中的所有地物，给出它们的种类、颜色、形状、大小等特征。
### 识别结果 ###
若图中出现了地物，请以 JSON 格式从左到右对它们进行描述，包括地物：种类、颜色、形状、大小、位置等。
"""

MODEL_DIR = "/root/models/FM9G4B-V"
IMG_DIR   = "/root/datasets/NWPU_VHR10/positive image set"  # 建议把数据放WSL本地
OUT_CSV   = "/root/projects/nwpu/preds.csv"

def extract_json_block(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: 
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype  = torch.bfloat16 if use_cuda else torch.float32
    print(f"device: {device}")

    # 加载模型/分词器：保持与你一致
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # 收集图片
    exts = (".jpg",".jpeg",".png",".tif",".tiff",".bmp")
    imgs = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]
    imgs.sort()
    print(f"found {len(imgs)} images under: {IMG_DIR}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 先把“原始输出”落盘，后面再慢慢加强解析
        w.writerow(["image_path", "raw_output", "parsed_json"])
        for p in imgs:
            try:
                image = Image.open(p).convert("RGB")
                msgs = [{'role': 'user', 'content': [image, PROMPT]}]

                # 单轮推理（关闭第二轮&流式，批量更快更稳）
                out = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    local_files_only=True,
                    do_sample=False,       # 输出更稳定，便于后处理
                    max_new_tokens=200,     # 控制长度，加速
                )
                parsed = extract_json_block(out)
                w.writerow([p, out, json.dumps(parsed, ensure_ascii=False)])
                print("OK:", os.path.basename(p))
            except Exception as e:
                print("FAIL:", p, e)

if __name__ == "__main__":
    main()
